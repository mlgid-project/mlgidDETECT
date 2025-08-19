"""
This module provides safe and efficient conversion utilities between CuPy arrays and OpenCV's `cv2.cuda_GpuMat`.

Purpose:
OpenCV's CUDA backend is powerful for GPU-accelerated image processing, but its Python API lacks direct support
for CuPy arrays in many versions. This module bridges that gap—enabling zero-copy GPU memory sharing when possible,
and falling back to CPU-based transfers only when necessary.

Features:
- Convert CuPy arrays to `cv2.cuda_GpuMat` using `createGpuMatFromCudaMemory` (OpenCV ≥ 4.8.0).
- Convert `cv2.cuda_GpuMat` back to CuPy arrays using the CUDA array interface (`cudaPtr()`).
- Graceful fallback to NumPy-based upload/download if the required GPU APIs are unavailable.

Why You Need This:
Without this module, converting between CuPy and OpenCV often involves downloading data to the CPU,
which is slow and defeats the purpose of GPU acceleration. This module ensures that:
- If your OpenCV build supports direct GPU memory access, it uses it.
- If not, it still works—just with a performance tradeoff.

Compatibility:
- CuPy
- OpenCV: full GPU support requires OpenCV ≥ 4.8.0 with CUDA enabled.
"""

import cv2
try:
    import cupy as cp
except ImportError:
    cp = None  # CuPy not available

# Check once at module level
_HAS_CREATE_GPU_MAT_FROM_CUDA = hasattr(cv2.cuda, "createGpuMatFromCudaMemory")
_HAS_CUDA_PTR = hasattr(cv2.cuda_GpuMat(), "cudaPtr")


def cv_cuda_gpumat_from_cp_array(arr) -> cv2.cuda.GpuMat:
    assert len(arr.shape) in (2, 3), "CuPy array must have 2 or 3 dimensions to be a valid GpuMat"

    if _HAS_CREATE_GPU_MAT_FROM_CUDA:

        type_map = {
            cp.dtype('uint8'): cv2.CV_8U,
            cp.dtype('int8'): cv2.CV_8S,
            cp.dtype('uint16'): cv2.CV_16U,
            cp.dtype('int16'): cv2.CV_16S,
            cp.dtype('int32'): cv2.CV_32S,
            cp.dtype('float32'): cv2.CV_32F,
            cp.dtype('float64'): cv2.CV_64F
        }
        depth = type_map.get(arr.dtype)
        assert depth is not None, "Unsupported CuPy array dtype"
        channels = 1 if len(arr.shape) == 2 else arr.shape[2]
        # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
        # (depth&7) + ((channels - 1) << 3)
        mat_type = depth + ((channels - 1) << 3)
        return cv2.cuda.createGpuMatFromCudaMemory(arr.__cuda_array_interface__['shape'][1::-1],
                                                mat_type,
                                                arr.__cuda_array_interface__['data'][0])
    else: 
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(arr.asnumpy())
        return gpu_mat




def safe_cv_cuda_gpumat_from_array(arr) -> cv2.cuda.GpuMat:
    try:
        if isinstance(arr, cp.ndarray):
            return cv_cuda_gpumat_from_cp_array(arr)
        elif isinstance(arr, np.ndarray):
            return cv2.cuda_GpuMat().upload(arr)
        else:
            raise TypeError("Input must be a CuPy or NumPy array")
    except AttributeError as e:
        print(f"[WARNING] CuPy to GpuMat conversion failed: {e}")
        print("[INFO] Falling back to NumPy array upload")
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
        return cv2.cuda_GpuMat().upload(arr)


def cp_array_from_cv_cuda_gpumat(mat: cv2.cuda.GpuMat):
    class CudaArrayInterface:
        def __init__(self, gpu_mat: cv2.cuda.GpuMat):
            w, h = gpu_mat.size()
            type_map = {
                cv2.CV_8U: "|u1",
                cv2.CV_8S: "|i1",
                cv2.CV_16U: "<u2", cv2.CV_16S: "<i2",
                cv2.CV_32S: "<i4",
                cv2.CV_32F: "<f4", cv2.CV_64F: "<f8",
            }
            self.__cuda_array_interface__ = {
                "version": 3,
                "shape": (h, w, gpu_mat.channels()) if gpu_mat.channels() > 1 else (h, w),
                "typestr": type_map[gpu_mat.depth()],
                "descr": [("", type_map[gpu_mat.depth()])],
                "stream": 1,
                "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()) if gpu_mat.channels() > 1
                else (gpu_mat.step, gpu_mat.elemSize()),
                "data": (gpu_mat.cudaPtr(), False),
            }

    if _HAS_CUDA_PTR:
        return cp.asarray(CudaArrayInterface(mat))
    else:
        # Fallback: download to NumPy and convert to CuPy
        np_arr = mat.download()
        return cp.asarray(np_arr)