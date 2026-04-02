import torch
import numpy as np
import copy
from typing import List, Tuple
from mlgiddetect.preprocessing.contrast_correction import unsharp_mask
from mlgiddetect.postprocessing.utils import filter_boxes, onnx_to_xyxy


def tta_inference(config, img_container, img_processing):
    """
    Test-time augmentation: combines predictions from original, horizontally flipped,
    and contrast-enhanced images using consensus.
    """
    imp = img_processing
    img_h, img_w = img_container.converted_polar_image.shape[-2], img_container.converted_polar_image.shape[-1]
    
    # Original inference results already in img_container
    original_boxes = img_container.boxes
    original_scores = img_container.scores
    
    # --- Horizontally flipped augmentation ---
    img_container_flipped = copy.deepcopy(img_container)
    img_container_flipped.converted_polar_image = np.flip(img_container_flipped.converted_polar_image, axis=-1).copy()
    
    raw_results_flipped = imp.infer(img_container_flipped)
    img_container_flipped = onnx_to_xyxy(config, img_container_flipped, raw_results_flipped)
    img_container_flipped = filter_boxes(config, img_container_flipped)
    
    # Flip boxes back to original coordinate system
    flipped_boxes = img_container_flipped.boxes.clone()
    flipped_boxes[:, 0] = img_w - img_container_flipped.boxes[:, 2]
    flipped_boxes[:, 2] = img_w - img_container_flipped.boxes[:, 0]
    flipped_scores = img_container_flipped.scores
    
    # --- Contrast-enhanced augmentation ---
    img_container_contrast = copy.deepcopy(img_container)
    img_container_contrast.converted_polar_image = unsharp_mask(
        img_container_contrast.converted_polar_image, sigma=1.5, strength=1.2
    )
    
    raw_results_contrast = imp.infer(img_container_contrast)
    img_container_contrast = onnx_to_xyxy(config, img_container_contrast, raw_results_contrast)
    img_container_contrast = filter_boxes(config, img_container_contrast)
    
    contrast_boxes = img_container_contrast.boxes
    contrast_scores = img_container_contrast.scores
    
    # --- Consensus combination ---
    all_boxes = [original_boxes, flipped_boxes, contrast_boxes]
    all_scores = [original_scores, flipped_scores, contrast_scores]
    
    fused_boxes, fused_scores = consensus_boxes(all_boxes, all_scores, iou_thr=0.2, min_sets=2)
    
    img_container.boxes = fused_boxes
    img_container.scores = fused_scores
    
    return img_container

def consensus_boxes(
    boxes_list: List[torch.Tensor],
    scores_list: List[torch.Tensor],
    iou_thr: float = 0.5,
    min_sets: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Consensus box fusion: keeps boxes that appear in >= min_sets predictions
    using IoU matching, and returns fused boxes with aggregated scores.
    """
    num_sets = len(boxes_list)
    device = boxes_list[0].device if boxes_list else torch.device('cpu')
    
    # Flatten all boxes and scores with set IDs
    all_boxes = []
    all_scores = []
    all_set_ids = []
    
    for set_id, (b, s) in enumerate(zip(boxes_list, scores_list)):
        if len(b) > 0:
            all_boxes.append(b)
            all_scores.append(s)
            all_set_ids.append(torch.full((len(b),), set_id, device=device, dtype=torch.long))
    
    if not all_boxes:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device)
    
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    set_ids = torch.cat(all_set_ids, dim=0)
    
    N = len(boxes)
    
    # Compute pairwise IoU
    iou_matrix = box_iou_matrix(boxes, boxes)
    
    # Greedy clustering
    remaining = torch.ones(N, dtype=torch.bool, device=device)
    fused_boxes = []
    fused_scores = []
    
    while remaining.any():
        # Get first remaining box
        remaining_idx = torch.where(remaining)[0]
        if len(remaining_idx) == 0:
            break
        idx = remaining_idx[0].item()
        
        # Find all overlapping boxes
        overlaps = (iou_matrix[idx] >= iou_thr) & remaining
        cluster_idx = torch.where(overlaps)[0]
        
        if len(cluster_idx) == 0:
            remaining[idx] = False
            continue
        
        # Get boxes in this cluster
        cluster_boxes = boxes[cluster_idx]
        cluster_scores = scores[cluster_idx]
        cluster_sets = set_ids[cluster_idx]
        
        # Check if cluster has enough diverse predictions
        unique_sets = torch.unique(cluster_sets)
        if len(unique_sets) >= min_sets:
            # Fuse boxes with score-weighted average
            weights = cluster_scores / cluster_scores.sum()
            fused_box = (cluster_boxes * weights.view(-1, 1)).sum(dim=0)
            fused_score = cluster_scores.mean() * (len(unique_sets) / num_sets)
            
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
        
        # Mark cluster as processed
        remaining[cluster_idx] = False
    
    if len(fused_boxes) == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device)
    
    return torch.stack(fused_boxes), torch.stack(fused_scores)

def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Vectorized IoU between two sets of boxes using torch
    boxes1: (N,4)
    boxes2: (M,4)
    """
    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / torch.clamp(union, min=1e-9)
    
    # Ensure diagonal is 1.0 (a box should have IoU 1.0 with itself)
    iou = torch.where(union == 0, torch.ones_like(iou), iou)
    
    return iou
