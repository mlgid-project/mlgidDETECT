import torch
from torchvision.ops import nms

# for output bounding box post-processing
def box_cxcywh_to_xyxy(config, x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
    boxes = torch.stack(b, dim=-1)[0]
    scale = torch.tensor([config.PREPROCESSING_POLAR_SHAPE[1], config.PREPROCESSING_POLAR_SHAPE[0], config.PREPROCESSING_POLAR_SHAPE[1], config.PREPROCESSING_POLAR_SHAPE[0]], dtype=boxes.dtype)
    boxes = boxes * scale
    return boxes

def onnx_to_xyxy(config, img_container, raw_results, num_select: int = 225):
    out_logits = torch.from_numpy(raw_results[0])
    out_bbox = torch.from_numpy(raw_results[1])

    prob = out_logits.sigmoid()
    num_select = min(num_select, prob.shape[1] * prob.shape[2])
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
    img_container.scores = topk_values[0]
    topk_boxes = topk_indexes[0] // out_logits.shape[2]
    # predicted class per selected box (used by class-aware NMS; harmless for single-class models)
    img_container.pred_labels = topk_indexes[0] % out_logits.shape[2]

    img_container.boxes = box_cxcywh_to_xyxy(config, out_bbox)
    img_container.boxes = img_container.boxes[topk_boxes]

    return img_container

def filter_boxes(config, img_container):
    """NMS then score threshold.

    If ``config.POSTPROCESSING_CLASSAWARE_NMS`` is set (2-class ring/segment model, segment=0 /
    ring=1), NMS is run per predicted class with class-specific IoU thresholds
    (ring -> POSTPROCESSING_NMSIOU_RING, segment -> POSTPROCESSING_NMSIOU_SEG). Otherwise a single
    NMS at POSTPROCESSING_NMSIOU is used (legacy single-class behaviour). Kept in lockstep with
    mlgidDETECT_DINO/util/postprocessing.py.
    """
    boxes, scores = img_container.boxes, img_container.scores
    labels = getattr(img_container, 'pred_labels', None)

    if getattr(config, 'POSTPROCESSING_CLASSAWARE_NMS', False) and labels is not None:
        class_iou = {1: config.POSTPROCESSING_NMSIOU_RING, 0: config.POSTPROCESSING_NMSIOU_SEG}
        keep_parts = []
        for cls, iou_thr in class_iou.items():
            cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
            if cls_idx.numel():
                kept = nms(boxes[cls_idx], scores[cls_idx], iou_thr)
                keep_parts.append(cls_idx[kept])
        idx_keep = torch.cat(keep_parts) if keep_parts else torch.empty(0, dtype=torch.long, device=boxes.device)
    else:
        idx_keep = nms(boxes, scores, config.POSTPROCESSING_NMSIOU)

    boxes = boxes[idx_keep]
    scores = scores[idx_keep]
    labels = labels[idx_keep] if labels is not None else None

    to_keep = scores > config.POSTPROCESSING_SCORE
    img_container.boxes = boxes[to_keep]
    img_container.scores = scores[to_keep]
    if labels is not None:
        img_container.pred_labels = labels[to_keep]

    return img_container

