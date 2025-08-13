import sys
import numpy as np
import cv2
import logging

try:
    import cupy as cp
except ImportError:
    cp = None  # CuPy not available

from mlgiddetect.preprocessing import (preprocess_geometry, contrast_correction, add_batch_and_color_channel,
                                        grayscale_to_color)

def standard_preprocessing(config, raw_reciprocal_img: np.array, counter = None):

    config.GEO_RECIPROCAL_SHAPE = list(raw_reciprocal_img.shape)

    if config.PREPROCESSING_CUDA:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(raw_reciprocal_img)
        raw_reciprocal_img = gpu_img
        
    #conversion to polar coordinates
    if config.PREPROCESSING_POLAR_CONVERSION:
        raw_polar_img = preprocess_geometry(config, raw_reciprocal_img)
    else:
        raw_polar_img = raw_reciprocal_img

    equalized_polar, mask = contrast_correction(config, raw_polar_img)
    equalized_polar = add_batch_and_color_channel(equalized_polar)
    mask = add_batch_and_color_channel(mask)

    #reshape for detr model
    if config.MODEL_TYPE == 'detr':
        equalized_polar = grayscale_to_color(equalized_polar)
        equalized_polar = equalized_polar[:,:,:,:]
        equalized_polar = np.pad(equalized_polar, ((0,0),(0,0,),(0,496), (0,0)))

    if config.PREPROCESSING_CUDA:
        equalized_polar = cp.asnumpy(equalized_polar)
        raw_polar_img = cp.asnumpy(raw_polar_img)
        mask = cp.asnumpy(mask)

    return equalized_polar, raw_polar_img, mask