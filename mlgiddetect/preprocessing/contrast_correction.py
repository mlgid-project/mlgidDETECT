import numpy as np
import cv2
try:
    import cupy as cp
except ImportError:
    cp = None  # CuPy not available

from mlgiddetect.configuration import Config
from mlgiddetect.utils import cv_cuda_gpumat_from_cp_array, cp_array_from_cv_cuda_gpumat
from scipy.ndimage import gaussian_filter

DEFAULT_CLAHE_LIMIT: float = 2000.
DEFAULT_CLAHE_COEF: float = 500.

# GPU CLAHE availability check (requires OpenCV built with CUDA and cudaimgproc module)
_HAS_CUDA_CLAHE = (
    cp is not None and
    hasattr(cv2, 'cuda') and
    hasattr(cv2.cuda, 'createCLAHE')
)

def normalize_with_std_mean(img: np.array, mean= None, std= None):
    if mean is None:
        mean = 0.485
    if std is None:
        std = 0.229
    return (img - mean) / std

def normalize(img, nonzero_indices):
    """Min-max normalize the masked region. Works with both NumPy and CuPy arrays."""
    xp = cp if (cp is not None and isinstance(img, cp.ndarray)) else np
    vals = img[nonzero_indices]
    return (vals - xp.nanmin(vals)) / (xp.nanmax(vals) - xp.nanmin(vals))


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
    """CPU CLAHE. Input must be a NumPy array; implicitly downloads CuPy arrays."""
    if cp is not None and isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
    return cv2.createCLAHE(clipLimit=limit, tileGridSize=(1, 1)).apply(img.astype('uint16')).astype(np.float32)

def _cuda_clahe_func(img_cp, limit: float = DEFAULT_CLAHE_LIMIT):
    """GPU CLAHE using OpenCV CUDA (zero CPU round-trip). Requires OpenCV with cudaimgproc."""
    img_uint16 = img_cp.astype(cp.uint16)
    gpu_mat = cv_cuda_gpumat_from_cp_array(img_uint16)
    clahe = cv2.cuda.createCLAHE(clipLimit=limit, tileGridSize=(1, 1))
    result_mat = clahe.apply(gpu_mat)
    return cp_array_from_cv_cuda_gpumat(result_mat).astype(cp.float32)

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
):

    if config.PREPROCESSING_CUDA:
        xp = cp
        img = cp_array_from_cv_cuda_gpumat(img)
    else:
        xp = np

    if config is not None:
        linear_normalization = config.PREPROCESSING_LINEAR_CONTRAST
        if config.PREPROCESSING_NO_CONTRASTCORRECTION:
            linear_normalization = False
            log = False
            clahe = False

    mask = ~xp.isnan(img) & (img != 0)

    if linear_normalization:
        upper_clip_limit = xp.percentile(img[mask],99.9)
        lower_clip_limit = xp.percentile(img[mask],5)

        img[mask] = xp.clip(img[mask], lower_clip_limit, upper_clip_limit)
        img[mask] =  normalize(img, mask)
        img = img *255

        if config.PREPROCESSING_CUDA:
            img = cv_cuda_gpumat_from_cp_array(img.astype(xp.uint8))
            img = cv2.cuda.equalizeHist(img)
            img = cp_array_from_cv_cuda_gpumat(img)
        else:
            img = cv2.equalizeHist(img.astype(xp.uint8))

        img = img /255
        img = img.astype(xp.float32)
        img[~mask] = 0
        return img, mask

    if log:
        img[mask] =  normalize(img, mask)
        img = xp.log10(img * coef + 1)

    if clahe:
        if config.PREPROCESSING_CUDA and _HAS_CUDA_CLAHE:
            img = _cuda_clahe_func(img * coef, limit)
        else:
            img = clahe_func(img * coef, limit)
        img[mask] = normalize(img, mask)
        img[~mask] = 0

    return img, mask

def contrast_correction(config, raw_polar_img: np.array):
    return _contrast_correction(config, raw_polar_img)


def add_batch_and_color_channel(img: np.array):
    img = np.repeat(img[ np.newaxis, :, :], 1, axis=0)
    return np.repeat(img[ np.newaxis, :, :], 1, axis=0)

def grayscale_to_color(img: np.array):
    return np.concatenate((img,)*3, axis=1)

def log_contrast(img_container):
    img = img_container.raw_polar_image
    if img.ndim == 2:
        img = img[np.newaxis, np.newaxis, :, :]
    elif img.ndim == 3:
        img = img[np.newaxis, :, :, :]

    eps = 1e-6
    out = np.log1p(img + eps)
    out = out / out.max()
    img_container.raw_polar_image = out.astype(np.float32)
    return img_container