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
from mlgiddetect.utils import cv_cuda_gpumat_from_cp_array
from . import flip_img_horizontal

DEFAULT_CLAHE_LIMIT: float = 2000.
DEFAULT_CLAHE_COEF: float = 500.
DEFAULT_POLAR_SHAPE: Tuple[int, int] = (512, 1024)
MAX_GRID_CACHE_SIZE: int = 5
DEFAULT_BEAM_CENTER: Tuple[float, float] = (0, 0)
DEFAULT_ALGORITHM: int = cv2.INTER_CUBIC

def _set_q_max(config: Config):
    config.GEO_QMAX = np.sqrt(((config.GEO_RECIPROCAL_SHAPE[0] / config.GEO_PIXELPERANGSTROEM) ** 2) + ((config.GEO_RECIPROCAL_SHAPE[1] / config.GEO_PIXELPERANGSTROEM) ** 2))

def get_q_max(config: Config):
    _set_q_max(config)
    return config.GEO_QMAX

def _get_quazipolar_grid(config, beam_center: Tuple[float, float] = DEFAULT_BEAM_CENTER,
                         shape: Tuple[int, int] = (10, 10),
                         polar_shape: Tuple[int, int] = DEFAULT_POLAR_SHAPE,
                         coef: float = 0.6,
                         ):


    if config.PREPROCESSING_CUDA:
        xp = cp
    else:
        xp = np

    y0, z0 = beam_center
    y = xp.arange(shape[1], dtype=xp.float32) - y0
    z = xp.arange(shape[0], dtype=xp.float32) - z0
    zz, yy = xp.meshgrid(z, y)  # meshgrid order: (z, y)

    rr = xp.sqrt(yy ** 2 + zz ** 2)
    phi = xp.arctan2(zz, yy)
    r_range = rr.min(), rr.max()
    phi_range = phi.min(), phi.max()

    phi = xp.linspace(*phi_range, polar_shape[0], dtype=xp.float32)
    r = xp.linspace(*r_range, polar_shape[1], dtype=xp.float32)

    r_matrix = xp.repeat(r[None, :], polar_shape[0], axis=0)
    p_matrix = xp.repeat(phi[:, None], polar_shape[1], axis=1)

    p_coef = coef / (1e-4 + r_matrix / r_matrix.max())
    p_matrix = xp.minimum(p_matrix * p_coef, xp.pi)

    polar_yy = r_matrix * xp.cos(p_matrix) + y0
    polar_zz = r_matrix * xp.sin(p_matrix) + z0

    if config.PREPROCESSING_CUDA:
        polar_yy = cv_cuda_gpumat_from_cp_array(polar_yy)
        polar_zz = cv_cuda_gpumat_from_cp_array(polar_zz)

    return polar_yy, polar_zz


def _get_polar_grid(config,
        img_shape: Tuple[int, int],
        polar_shape: Tuple[int, int],
        beam_center: Tuple[float, float],
):

    if config.PREPROCESSING_CUDA:
        xp = cp
    else:
        xp = np

    y0, z0 = beam_center

    y = (xp.arange(img_shape[1]) - y0)
    z = (xp.arange(img_shape[0]) - z0)

    yy, zz = xp.meshgrid(y, z)
    rr = xp.sqrt(yy ** 2 + zz ** 2)
    phi = xp.arctan2(zz, yy)
    r_range = (rr.min(), rr.max())
    phi_range = phi.min(), phi.max()

    phi = xp.linspace(*phi_range, polar_shape[0])
    r = xp.linspace(*r_range, polar_shape[1])

    r_matrix = r[xp.newaxis, :].repeat(polar_shape[0], axis=0)
    p_matrix = phi[:, xp.newaxis].repeat(polar_shape[1], axis=1)

    polar_yy = r_matrix * xp.cos(p_matrix) + y0
    polar_zz = r_matrix * xp.sin(p_matrix) + z0

    if config.PREPROCESSING_CUDA:
        polar_yy = cv_cuda_gpumat_from_cp_array(polar_yy)
        polar_zz = cv_cuda_gpumat_from_cp_array(polar_zz)


    return polar_yy, polar_zz


def _calc_polar_img(config, img: np.ndarray, yy: np.ndarray, zz: np.ndarray, algorithm: int) -> np.ndarray or None:
    try:
        if config.PREPROCESSING_CUDA:
            return cv2.cuda.remap(img,
                            yy,
                            zz,
                            interpolation=algorithm)
        else:
            return cv2.remap(img.astype(np.float32),
                yy.astype(np.float32),
                zz.astype(np.float32),
                interpolation=algorithm)

    except cv2.error:
        logging.error("Error in polar conversion!")
        sys.exit()


def calc_quazipolar_image(config, img: np.ndarray,
                          beam_center: Tuple[float, float] = DEFAULT_BEAM_CENTER,
                          polar_shape: Tuple[int, int] = DEFAULT_POLAR_SHAPE,
                          algorithm=cv2.INTER_LINEAR, coef: float = 0.6) -> np.ndarray or None:

    # Detect if input is CUDA or numpy and extract shape
    if isinstance(img, cv2.cuda_GpuMat):
        height, width = img.size()
        img_shape = (height, width)
    elif isinstance(img, np.ndarray):
        img_shape = img.shape

    yy, zz = _get_quazipolar_grid(config, beam_center, img_shape, polar_shape, coef=coef)

    return _calc_polar_img(config, img, yy, zz, algorithm)

def calc_polar_image(config,
        img: np.ndarray,
        polar_shape: Tuple[int, int] = DEFAULT_POLAR_SHAPE,
        beam_center: Tuple[float, float] = DEFAULT_BEAM_CENTER,
        algorithm: int = DEFAULT_ALGORITHM,
) -> np.ndarray or None:
    yy, zz = _get_polar_grid(config, img.shape, polar_shape, beam_center)

    return _calc_polar_img(config, img, yy, zz, algorithm)


def preprocess_geometry(config, raw_reciprocal_img: np.array):
    get_q_max(config)
    if config.PREPROCESSING_FLIPHORIZONTAL:
        raw_reciprocal_img = flip_img_horizontal(raw_reciprocal_img)
    if config.PREPROCESSING_QUAZIPOLAR:
        return calc_quazipolar_image(config, raw_reciprocal_img, polar_shape=config.PREPROCESSING_POLAR_SHAPE)
    else:
        return calc_polar_image(config, raw_reciprocal_img, polar_shape=config.PREPROCESSING_POLAR_SHAPE)