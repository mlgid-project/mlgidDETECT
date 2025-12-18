import sys
import logging
from typing import Tuple
import numpy as np
import cv2
try:
    import cupy as cp
except ImportError:
    cp = None  # CuPy not available

from mlgiddetect.configuration import Config
from mlgiddetect.utils import cv_cuda_gpumat_from_cp_array, cp_array_from_cv_cuda_gpumat

DEFAULT_CLAHE_LIMIT: float = 2000.
DEFAULT_CLAHE_COEF: float = 500.

def normalize_with_std_mean(img: np.array, mean= None, std= None):
    if mean is None:
        mean = 0.485
    if std is None:
        std = 0.229
    return (img - mean) / std

def normalize(img: np.ndarray, nonzero_indices: np.ndarray) -> np.ndarray:
    return (img[nonzero_indices] - np.nanmin(img[nonzero_indices])) / (np.nanmax(img[nonzero_indices]) - np.nanmin(img[nonzero_indices]))


def normalize_image(image, mean=0.5, std=0.1):
    # Convert image to float32
    image = image.astype(np.float32)
    
    # Normalize to range [0, 1], ignoring zeros
    nonzero_indices = image != 0    
    #image = (image - image.min()) / (image.max() - image.min())
    nonzero_values = image[nonzero_indices]
    min_nonzero = np.min(nonzero_values)
    max_nonzero = np.max(nonzero_values)
    image[nonzero_indices] = (image[nonzero_indices] - min_nonzero) / (max_nonzero - min_nonzero)
    
    # Normalize to desired mean and std
    image[nonzero_indices] = image[nonzero_indices] * std + (mean-(std/2))
    
    return image

def clahe_func(img, limit: float = DEFAULT_CLAHE_LIMIT):

    return cv2.createCLAHE(clipLimit=limit, tileGridSize=(1, 1)).apply(img.astype('uint16')).astype(np.float32)

def equalize_hist(config, img):

    if config.PREPROCESSING_CUDA:
        xp = cp
    else:
        xp = np
    # Ensure the image is in uint8 format
    img = img.astype(xp.uint8)

    # Compute histogram
    hist, bins = xp.histogram(img.flatten(), 256, [0,256])

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_masked = xp.ma.masked_equal(cdf, 0)  # Mask zeros to avoid division by zero

    # Normalize the CDF
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    cdf_normalized = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
    cdf_final = xp.ma.filled(cdf_normalized, 0).astype(xp.uint8)

    # Map the original image pixels through the equalized CDF
    return cdf_final[img]

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def _contrast_correction(
        config: Config = None,    
        img: np.array = None, 
        limit: float = DEFAULT_CLAHE_LIMIT,
        coef: float = DEFAULT_CLAHE_COEF,
        clahe: bool = True,
        log: bool = True,
        linear_normalization = False,
        linear_perc_997 = False
):

    if config.PREPROCESSING_CUDA:
        xp = cp
        img = cp_array_from_cv_cuda_gpumat(img)
    else:
        xp = np

    if config is not None:
        linear_normalization = config.PREPROCESSING_LINEAR_CONTRAST
        linear_perc_997 = config.PREPROCESSING_LINEAR_PERC_977
        log = config.PREPROCESSING_LOGARITHM
        clahe = not config.PREPROCESSING_LINEAR_CONTRAST
        if config.PREPROCESSING_NO_CONTRASTCORRECTION:
            linear_normalization = False
            log = False
            clahe = False

    clahe = True
    log = True
    linear_normalization = False
    mask = ~xp.isnan(img) & (img != 0)

    if log:
        img = np.nan_to_num(img)        
        img = xp.log(img+1)
        img = xp.log(img+1)
        img[mask] =  normalize(img, mask)
        #return img.astype(np.float32), mask

    if linear_normalization:
        # Ensure float32
        img = img.astype(np.float32)

        #if linear_perc_997:
            # Percentile clipping
        upper_clip_limit = np.percentile(img[mask], 80.0 if linear_perc_997 else 99)
        lower_clip_limit = np.percentile(img[mask], 5.0)

        # Clip and normalize
        img[mask] = np.clip(img[mask], lower_clip_limit, upper_clip_limit)
        img[mask] = (img[mask] - lower_clip_limit) / (upper_clip_limit - lower_clip_limit)
        #img[mask] = np.clip(img[mask], 0, 1)

        ###############
        img = np.nan_to_num(img)        
        #img = xp.log(img+1)
        img[mask] =  normalize(img, mask)
        img[~mask] = 0.0
        #return img.astype(np.float32), mask
        ########

        # Histogram equalization (float32 version)
        hist, bins = np.histogram(img[mask], bins=256, range=(0.0, 1.0))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize to [0,1]

        # Create LUT and apply
        bin_centers = (bins[:-1] + bins[1:]) / 2
        lut = np.interp(img[mask], bin_centers, cdf)
        img[mask] = lut

        # Final scaling and cleanup
        img[~mask] = 0.0
        return img.astype(np.float32), mask


    if clahe:
        img = clahe_func(img * coef, limit)
        img[mask] = normalize(img,mask)
        img[~mask] = 0

    return img, mask

def contrast_correction(config, raw_polar_img: np.array):
    return _contrast_correction(config, raw_polar_img)


def add_batch_and_color_channel(img: np.array):
    img = np.repeat(img[ np.newaxis, :, :], 1, axis=0)
    return np.repeat(img[ np.newaxis, :, :], 1, axis=0)

def grayscale_to_color(img: np.array):
    return np.concatenate((img,)*3, axis=1)
