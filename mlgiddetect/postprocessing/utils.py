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