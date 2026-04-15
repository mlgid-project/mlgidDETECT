import torch
from torchvision.ops import nms
from typing import List, Tuple

# for output bounding box post-processing
def box_cxcywh_to_xyxy(config, x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
    boxes = torch.stack(b, dim=-1)[0]
    scale = torch.tensor([config.PREPROCESSING_POLAR_SHAPE[1], config.PREPROCESSING_POLAR_SHAPE[0], config.PREPROCESSING_POLAR_SHAPE[1], config.PREPROCESSING_POLAR_SHAPE[0]], dtype=boxes.dtype)
    boxes = boxes * scale
    return boxes

def filter_non_elong(img_container):
    y_extent = img_container.boxes[:,3] - img_container.boxes[:,1]
    x_extent = img_container.boxes[:,2] - img_container.boxes[:,0]
    keep = x_extent*1.15 < y_extent
 
    img_container.scores = img_container.scores[keep]
    img_container.boxes = img_container.boxes[keep]
    return img_container

def onnx_to_xyxy(config, img_container, raw_results, num_select: int = 150):
    out_logits = torch.from_numpy(raw_results[0])
    out_bbox = torch.from_numpy(raw_results[1])

    prob = out_logits.sigmoid()
    num_select = min(num_select, prob.shape[1] * prob.shape[2])
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
    img_container.scores = topk_values[0]
    topk_boxes = topk_indexes[0] // out_logits.shape[2]
    
    img_container.boxes = box_cxcywh_to_xyxy(config, out_bbox)
    img_container.boxes = img_container.boxes[topk_boxes]

    return img_container

def filter_boxes(config, img_container):
    img_container = filter_non_elong(img_container)

    idx_keep = nms(img_container.boxes, img_container.scores, config.POSTPROCESSING_NMSIOU)
    boxes = img_container.boxes[idx_keep]
    scores = img_container.scores[idx_keep]

    to_keep = scores > config.POSTPROCESSING_SCORE
    img_container.boxes = boxes[to_keep]
    img_container.scores = scores[to_keep]

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

def box_flip_horizontal(img_container):
    flipped_boxes = img_container.boxes.clone()
    flipped_boxes[:, 0] = img_container.raw_polar_image.shape[-1] - img_container.boxes[:, 2]
    flipped_boxes[:, 2] = img_container.raw_polar_image.shape[-1] - img_container.boxes[:, 0]
    img_container.boxes = flipped_boxes
    return img_container