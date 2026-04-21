import numpy as np
import copy
from mlgiddetect.postprocessing.postprocessing import standard_postprocessing
from mlgiddetect.preprocessing.contrast_correction import log_contrast
from mlgiddetect.postprocessing.utils import filter_boxes, onnx_to_xyxy, consensus_boxes, box_flip_horizontal


def tta_inference(config, img_container, img_processing):
    """
    Test-time augmentation: combines predictions from original, horizontally flipped,
    and contrast-enhanced images using consensus.
    """
    imp = img_processing
    
    # --- Horizontally flipped augmentation ---
    img_container_flipped = copy.deepcopy(img_container)
    img_container_flipped.converted_polar_image = np.flip(img_container_flipped.converted_polar_image, axis=-1).copy()
    
    # --- Perform inference on flipped image ---
    raw_results_flipped = imp.infer(img_container_flipped)
    img_container_flipped = standard_postprocessing(img_container_flipped, raw_results_flipped)
    
    # --- Flip boxes back to original coordinate system ---
    img_container_flipped = box_flip_horizontal(img_container_flipped)
    
    # --- Contrast-enhanced augmentation ---
    img_container_contrast = copy.deepcopy(img_container)
    img_container_contrast = log_contrast(img_container_contrast)
    
    # --- Perform inference on contrast-enhanced image ---
    raw_results_contrast = imp.infer(img_container_contrast, use_raw=True)
    img_container_contrast = standard_postprocessing(img_container_contrast, raw_results_contrast)
    
    # --- Consensus combination ---
    all_boxes = [img_container.boxes, img_container_flipped.boxes, img_container_contrast.boxes]
    all_scores = [img_container.scores, img_container_flipped.scores, img_container_contrast.scores]
    
    img_container.boxes, img_container.scores = consensus_boxes(all_boxes, all_scores, iou_thr=0.2, min_sets=2)
    
    return img_container
