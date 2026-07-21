import os
import sys
import logging
import numpy as np
import onnxruntime as rt
from mlgiddetect.utils import path_utils
from mlgiddetect.dataloader import ImageContainer

class Inference:
    def __init__(self, config, model_path=None):
        self.config = config
        if model_path is None:
            model_path = path_utils.get_model_path(config)
        if model_path is None:
            logging.error('could not get model file. Exiting')
            sys.exit()

        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3
        # Set intra-op threads explicitly so onnxruntime does NOT auto-pin CPU affinity
        # (pthread_setaffinity_np fails + floods stderr on cgroup-restricted SLURM nodes).
        sess_options.intra_op_num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
        logging.info("Loading model")
        available_providers = rt.get_available_providers()
        # onnxruntime GPU inference does NOT depend on torch's CUDA build, so we do not gate on
        # torch.cuda.is_available() (the conda env may ship a CPU-only torch while onnxruntime-gpu
        # + the CUDA runtime are present). Gate only on the provider being available.
        use_gpu = (
            "CUDAExecutionProvider" in available_providers and
            not config.MODEL_FORCE_CPU
        )

        preferred_providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        if preferred_providers[0] == 'CUDAExecutionProvider':
            logging.info("Using the GPU for inference")
            sess_options.intra_op_num_threads = 1
        self.sess = rt.InferenceSession(model_path, providers=preferred_providers, sess_options=sess_options)

    def infer(self, img_container: ImageContainer, use_raw=False):
        # Run inference with ONNX Runtime
        input_name = self.sess.get_inputs()[0].name

        try:
            if use_raw:
                return self.sess.run(None, {input_name: img_container.raw_polar_image.astype(np.float32)})
            else:
                return self.sess.run(None, {input_name: img_container.converted_polar_image.astype(np.float32)})
        except rt.capi.onnxruntime_pybind11_state.RuntimeException as e:
            error_message = str(e)
            if "Failed to allocate memory" in error_message or "BFCArena::AllocateRawInternal" in error_message:
                logging.error("GPU memory allocation failed. Consider using CPU execution with the option FORCE_CPU = True")
                # Optionally, re-raise or handle gracefully
                raise MemoryError("GPU memory exhausted during inference.") from e
            else:
                # Re-raise unexpected RuntimeExceptions
                raise


def _resolve_onnx(config, value):
    """Map an ONNX config value to a model path.

    'base'                 -> the default model mlgidDETECT downloads for MODEL_TYPE
                              (path_utils.get_model_path).
    a MODEL_URLS keyword   -> that named downloadable model, e.g. 'dino_old' (the legacy
                              91-class model) or a future 'ssl_pretrain'; cached as <keyword>.onnx.
    anything else          -> treated as a path to an .onnx file.
    """
    if value == 'base':
        return path_utils.get_model_path(config)
    if value in path_utils.MODEL_URLS:
        return path_utils.get_named_model_path(config, value)
    return value


def load_sessions(config):
    """Build the inference session(s) for this run.

    MODEL.ONNX_BASE is the model used on its own; MODEL.ONNX_ENSEMBLE is fused with it when
    MODEL.ENSEMBLE_ENABLED is True (dino only). Each value is either a path to an .onnx file
    or the keyword 'base' (the auto-downloaded default model for MODEL_TYPE).

    dino + ENSEMBLE_ENABLED -> [ONNX_BASE, ONNX_ENSEMBLE]  (fused by ensemble_postprocessing)
    otherwise               -> [ONNX_BASE]                 (faster_rcnn always lands here)
    """
    base = _resolve_onnx(config, config.MODEL_ONNX_BASE)
    if config.MODEL_TYPE == 'dino' and config.MODEL_ENSEMBLE_ENABLED:
        if config.MODEL_ONNX_ENSEMBLE is None:
            logging.warning('MODEL.ENSEMBLE_ENABLED is True but MODEL.ONNX_ENSEMBLE is unset; '
                            'running ONNX_BASE alone.')
            return [Inference(config, model_path=base)]
        ensemble = _resolve_onnx(config, config.MODEL_ONNX_ENSEMBLE)
        return [Inference(config, model_path=base), Inference(config, model_path=ensemble)]
    return [Inference(config, model_path=base)]