import logging
import torch
import numpy as np

from mlgiddetect.postprocessing import rescale_bboxes, boxes_polar_to_reciprocal, boxes_reciprocal_q_to_xy, SmallQFilter, MergeBoxesPostprocessing, StandardPostprocessing, polar_to_cartesian
from torchvision.ops import nms

def standard_postprocessing(img_container, raw_results):
    if raw_results[0].size == 0:
        logging.error('No peaks found in image %s. Maybe try another contrast correction.', img_container.nr)
        return img_container

    config = img_container.config
    if config.MODEL_TYPE == 'detr':
        img_container.scores = raw_results[0][0]
        img_container.is_ring = raw_results[1][0] == 2
        img_container.boxes = rescale_bboxes(config,torch.tensor(raw_results[2][0]))

        topk = torch.topk(torch.tensor(img_container.scores), k=min(150, img_container.scores.size))
        img_container.boxes = img_container.boxes[topk.indices]
        img_container.scores = img_container.scores[topk.indices]
        img_container.is_ring =img_container.is_ring[topk.indices]

        idx_keep = nms(torch.tensor(img_container.boxes), torch.tensor(img_container.scores), 0.3)
        img_container.boxes = img_container.boxes[idx_keep]
        img_container.scores = img_container.scores[idx_keep]
        img_container.is_ring =img_container.is_ring[idx_keep]

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