import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

def flip_img_horizontal(image: np.array):
    return np.flipud(image)

def add_batch_and_color_channel(img):
    """Add batch and channel dimensions. Works with both NumPy and CuPy arrays."""
    xp = cp if (cp is not None and isinstance(img, cp.ndarray)) else np
    img = xp.repeat(img[xp.newaxis, :, :], 1, axis=0)
    return xp.repeat(img[xp.newaxis, :, :], 1, axis=0)
