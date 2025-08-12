import logging
import torch

from mlgiddetect.postprocessing import rescale_bboxes, boxes_polar_to_reciprocal, boxes_reciprocal_q_to_xy, SmallQFilter, MergeBoxesPostprocessing, StandardPostprocessing, polar_to_cartesian

def standard_postprocessing(img_container, raw_results):
    if raw_results[0].size == 0:
        logging.error('No peaks found in image %s. Maybe try another contrast correction.', img_container.nr)
        return img_container

    config = img_container.config
    if config.MODEL_TYPE == 'detr':
        img_container.scores = torch.tensor(raw_results[1][0]).softmax(-1)[:,1].numpy()
        boxes = raw_results[0][0]
        boxes = rescale_bboxes(config,torch.tensor(boxes))        
        boxes[:,0] = boxes[:,0]-16
        boxes[:,2] = boxes[:,2]-16
        boxes[:,1] = boxes[:,1]-8
        boxes[:,3] = boxes[:,3]-8
        img_container.boxes = boxes     



    if config.MODEL_TYPE == 'faster_rcnn':
        postprocessing = (
                SmallQFilter(50) +
                MergeBoxesPostprocessing(min_score=0.01) +
                StandardPostprocessing(nms_level=0.01, score_level=0.01)
        )
        img_container.boxes, img_container.scores = postprocessing(torch.tensor(raw_results[0]), torch.tensor(raw_results[1]))

    reciprocal_boxes_q = boxes_polar_to_reciprocal(config, img_container.boxes)
    img_container = boxes_reciprocal_q_to_xy(img_container, reciprocal_boxes_q)
    img_container = polar_to_cartesian(img_container)

    return img_container