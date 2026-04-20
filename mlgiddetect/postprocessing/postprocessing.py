import logging
import torch
from mlgiddetect.postprocessing.utils import filter_boxes, onnx_to_xyxy
from mlgiddetect.postprocessing import boxes_polar_to_reciprocal, boxes_reciprocal_q_to_xy, SmallQFilter, MergeBoxesPostprocessing, StandardPostprocessing, polar_to_cartesian
from mlgiddetect.inference.tta_inference import tta_inference

# Build once at import time — constructing this chain per image is unnecessary overhead
_FASTER_RCNN_POSTPROCESSING = (
    SmallQFilter(50) +
    MergeBoxesPostprocessing(min_score=0.01) +
    StandardPostprocessing(nms_level=0.01, score_level=0.01)
)

def standard_postprocessing(img_container, raw_results, img_processingimp):
    if raw_results[0].size == 0:
        logging.error('No peaks found in image %s. Maybe try another contrast correction.', img_container.nr)
        return img_container

    config = img_container.config
    if config.MODEL_TYPE == 'dino':
        img_container = onnx_to_xyxy(config, img_container, raw_results)
        img_container = filter_boxes(config, img_container) 
        if config.POSTPROCESSING_TTA: 
            img_container = tta_inference(config, img_container, img_processingimp)

    if config.MODEL_TYPE == 'faster_rcnn':
        img_container.boxes, img_container.scores = _FASTER_RCNN_POSTPROCESSING(torch.tensor(raw_results[0]), torch.tensor(raw_results[1]))
        img_container.is_ring = [0] * len(img_container.boxes)
    reciprocal_boxes_q = boxes_polar_to_reciprocal(config, img_container.boxes)
    img_container = boxes_reciprocal_q_to_xy(config, img_container, reciprocal_boxes_q)
    img_container = polar_to_cartesian(img_container)

    return img_container