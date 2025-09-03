import logging
import torch
import numpy as np

from mlgiddetect.postprocessing import rescale_bboxes, boxes_polar_to_reciprocal, boxes_reciprocal_q_to_xy, SmallQFilter, MergeBoxesPostprocessing, StandardPostprocessing, polar_to_cartesian

def standard_postprocessing(img_container, raw_results):
    if raw_results[0].size == 0:
        logging.error('No peaks found in image %s. Maybe try another contrast correction.', img_container.nr)
        return img_container

    config = img_container.config
    if config.MODEL_TYPE == 'detr':
        img_container.scores = torch.tensor(raw_results[1][0]).softmax(-1)[:,1].numpy()

        scores_rings = torch.tensor(raw_results[1][0]).softmax(-1)[:,2].numpy()
        scores_peaks = torch.tensor(raw_results[1][0]).softmax(-1)[:,1].numpy()
        to_keep = np.logical_or(scores_rings > config.POSTPROCESSING_SCORELIMIT_RINGS ,scores_peaks > config.POSTPROCESSING_SCORELIMIT_PEAKS )
        img_container.scores = np.maximum(scores_rings , scores_peaks)[to_keep]
        img_container.is_ring = (scores_rings > config.POSTPROCESSING_SCORELIMIT_RINGS)[to_keep]
        boxes = raw_results[0][0]
        boxes = boxes[to_keep]
        boxes = rescale_bboxes(config,torch.tensor(boxes))
        img_container.boxes = boxes     

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