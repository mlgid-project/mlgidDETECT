import copy
import numpy as np
import torch

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def shrink_y(boxes, f=2.):
    c = (boxes[:,1] + boxes[:,3]) / 2
    h = (boxes[:,3] - boxes[:,1]) / (2*f)
    boxes[:,1], boxes[:,3] = c - h, c + h
    return boxes

def rescale_bboxes(config, out_bbox):
    size = copy.deepcopy(config.PREPROCESSING_POLAR_SHAPE)
    img_h, img_w = size
    img_h = img_h+832
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    b = shrink_y(b)
    return b