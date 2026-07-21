import logging
import torch
from mlgiddetect.postprocessing.utils import filter_boxes, onnx_to_xyxy
from mlgiddetect.postprocessing import boxes_polar_to_reciprocal, boxes_reciprocal_q_to_xy, SmallQFilter, MergeBoxesPostprocessing, StandardPostprocessing, polar_to_cartesian

def standard_postprocessing(img_container, raw_results):
    if raw_results[0].size == 0:
        logging.error('No peaks found in image %s. Maybe try another contrast correction.', img_container.nr)
        return img_container

    config = img_container.config
    if config.MODEL_TYPE == 'dino':
        img_container = onnx_to_xyxy(config, img_container, raw_results)
        img_container = filter_boxes(config, img_container)

    if config.MODEL_TYPE == 'faster_rcnn':
        postprocessing = (
                SmallQFilter(50) +
                MergeBoxesPostprocessing(min_score=0.01) +
                StandardPostprocessing(nms_level=0.01, score_level=0.01)
        )
        img_container.boxes, img_container.scores = postprocessing(torch.tensor(raw_results[0]), torch.tensor(raw_results[1]))
        img_container.is_ring = [0] * len(img_container.boxes)
    reciprocal_boxes_q = boxes_polar_to_reciprocal(config, img_container.boxes)
    img_container = boxes_reciprocal_q_to_xy(config, img_container, reciprocal_boxes_q)
    img_container = polar_to_cartesian(img_container)

    return img_container


def _ensemble_topk(config, img_container, raw_results):
    """dino onnx_to_xyxy (top-k cxcywh->xyxy) for one model; return its detection pool."""
    onnx_to_xyxy(config, img_container, raw_results)
    return (img_container.boxes.clone(), img_container.scores.clone(),
            img_container.pred_labels.clone())


def ensemble_postprocessing(img_container, raw_results_list):
    """Detection-level fusion of >=2 dino models (mirrors mlgidDETECT_DINO/ensemble_eval.py).

    Pools each model's top-k detections, then runs the SAME class-aware NMS (filter_boxes) over
    the union, then the standard q-space coordinate transforms. The two models (ssl1 + baseline)
    are complementary, so pooling at full confidence (NMS) beats agreement-weighted fusion (WBF).
    """
    if any(r[0].size == 0 for r in raw_results_list):
        logging.error('No peaks found in image %s.', img_container.nr)
        return img_container
    config = img_container.config
    assert config.MODEL_TYPE == 'dino', 'ensemble_postprocessing supports MODEL_TYPE=dino only'

    boxes, scores, labels = [], [], []
    for raw in raw_results_list:
        b, s, l = _ensemble_topk(config, img_container, raw)
        boxes.append(b); scores.append(s); labels.append(l)
    img_container.boxes = torch.cat(boxes)
    img_container.scores = torch.cat(scores)
    img_container.pred_labels = torch.cat(labels)

    img_container = filter_boxes(config, img_container)          # class-aware NMS over the pool
    reciprocal_boxes_q = boxes_polar_to_reciprocal(config, img_container.boxes)
    img_container = boxes_reciprocal_q_to_xy(config, img_container, reciprocal_boxes_q)
    img_container = polar_to_cartesian(img_container)
    return img_container