import torch
from torchvision.ops import nms, box_iou
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

def perform_nms(config, img_container):
    # Distinguish between rings and segments based on vertical coverage
    img_height = config.PREPROCESSING_POLAR_SHAPE[0]
    y_extent = img_container.boxes[:, 3] - img_container.boxes[:, 1]
    is_ring = y_extent >= img_height * 0.35 # heuristic: rings cover at least 35% of vertical axis, segments less
    
    # Separate boxes into rings and segments
    ring_mask = is_ring
    segment_mask = ~is_ring
    
    ring_boxes = img_container.boxes[ring_mask]
    ring_scores = img_container.scores[ring_mask]
    segment_boxes = img_container.boxes[segment_mask]
    segment_scores = img_container.scores[segment_mask]
    
    # Apply NMS to rings with lenient threshold
    ring_idx_keep = []
    if len(ring_boxes) > 0:
        ring_idx_keep = nms(ring_boxes, ring_scores, iou_threshold=config.POSTPROCESSING_NMSRING)
    
    # Apply NMS to segments with strict threshold
    segment_idx_keep = []
    if len(segment_boxes) > 0:
        segment_idx_keep = nms(segment_boxes, segment_scores, iou_threshold=config.POSTPROCESSING_NMSSEGMENT)
    
    # Reconstruct boxes and scores
    filtered_boxes = []
    filtered_scores = []
    
    if len(ring_idx_keep) > 0:
        filtered_boxes.append(ring_boxes[ring_idx_keep])
        filtered_scores.append(ring_scores[ring_idx_keep])

    if len(segment_idx_keep) > 0:
        filtered_boxes.append(segment_boxes[segment_idx_keep])
        filtered_scores.append(segment_scores[segment_idx_keep])
    
    if filtered_boxes:
        img_container.boxes = torch.cat(filtered_boxes)
        img_container.scores = torch.cat(filtered_scores)
    else:
        img_container.boxes = torch.empty((0, 4), device=img_container.boxes.device)
        img_container.scores = torch.empty((0,), device=img_container.scores.device)

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
    #img_container = filter_non_elong(img_container)
    
    perform_nms(config, img_container)
    
    # Apply score threshold
    to_keep = img_container.scores > config.POSTPROCESSING_SCORE
    img_container.boxes = img_container.boxes[to_keep]
    img_container.scores = img_container.scores[to_keep]

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

    all_boxes, all_scores, all_set_ids = [], [], []
    for set_id, (b, s) in enumerate(zip(boxes_list, scores_list)):
        if len(b) > 0:
            all_boxes.append(b)
            all_scores.append(s)
            all_set_ids.append(torch.full((len(b),), set_id, device=device, dtype=torch.long))

    if not all_boxes:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device)

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    set_ids = torch.cat(all_set_ids)

    # anchor each cluster to its highest-score box
    order = scores.argsort(descending=True)
    boxes, scores, set_ids = boxes[order], scores[order], set_ids[order]

    iou_matrix = box_iou(boxes, boxes)
    processed = torch.zeros(len(boxes), dtype=torch.bool, device=device)
    fused_boxes, fused_scores = [], []

    for i in range(len(boxes)):
        if processed[i]:
            continue
        cluster_mask = (iou_matrix[i] >= iou_thr) & ~processed
        cluster_idx = cluster_mask.nonzero(as_tuple=True)[0]

        cluster_sets = set_ids[cluster_idx]
        unique_sets = cluster_sets.unique()
        if unique_sets.numel() >= min_sets:
            cluster_boxes = boxes[cluster_idx]
            cluster_scores = scores[cluster_idx]
            weights = cluster_scores / cluster_scores.sum()
            fused_boxes.append((cluster_boxes * weights.unsqueeze(1)).sum(0))
            fused_scores.append(cluster_scores.mean() * (unique_sets.numel() / num_sets))

        processed[cluster_mask] = True

    if not fused_boxes:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device)

    return torch.stack(fused_boxes), torch.stack(fused_scores)


def box_flip_horizontal(img_container):
    flipped_boxes = img_container.boxes.clone()
    flipped_boxes[:, 0] = img_container.raw_polar_image.shape[-1] - img_container.boxes[:, 2]
    flipped_boxes[:, 2] = img_container.raw_polar_image.shape[-1] - img_container.boxes[:, 0]
    img_container.boxes = flipped_boxes
    return img_container