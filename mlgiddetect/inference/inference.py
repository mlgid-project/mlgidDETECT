import sys
import logging
import numpy as np
import onnxruntime as rt
from mlgiddetect.utils import path_utils
from mlgiddetect.dataloader import ImageContainer
import torch

class Inference:
    def __init__(self, config):
        self.config = config
        model_path = path_utils.get_model_path(config)
        if model_path is None:
            logging.error('could not get model file. Exiting')
            sys.exit()

        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3
        logging.info("Loading model")
        available_providers = rt.get_available_providers()
        use_gpu = (
            "CUDAExecutionProvider" in available_providers and
            torch.cuda.is_available() and
            not config.MODEL_FORCE_CPU
        )

        preferred_providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        if preferred_providers[0] == 'CUDAExecutionProvider':
            logging.info("Using the GPU for inference")
            sess_options.intra_op_num_threads = 1
        self.sess = rt.InferenceSession(model_path, providers=preferred_providers, sess_options=sess_options)

    def infer(self, img_container: ImageContainer):
        # Run inference with ONNX Runtime
        input_name = self.sess.get_inputs()[0].name

        try:
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
