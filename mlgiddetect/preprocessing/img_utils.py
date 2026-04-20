import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

def flip_img_horizontal(image: np.array):
    return np.flipud(image)

def add_batch_and_color_channel(img):
    """Add batch and channel dimensions as views (no copy). Works with NumPy and CuPy arrays."""
    return img[np.newaxis, np.newaxis, :, :] if not (cp is not None and isinstance(img, cp.ndarray)) else img[cp.newaxis, cp.newaxis, :, :]
