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
        self.use_cuda = use_gpu
        self.device_id = 0

        if preferred_providers[0] == 'CUDAExecutionProvider':
            logging.info("Using the GPU for inference")
            sess_options.intra_op_num_threads = 1
        self.sess = rt.InferenceSession(model_path, providers=preferred_providers, sess_options=sess_options)

    def infer(self, img_container: ImageContainer, use_raw=False):
        # Run inference with ONNX Runtime
        input_name = self.sess.get_inputs()[0].name

        try:
            if use_raw:
                img_np = img_container.raw_polar_image.astype(np.float32)
            else:
                img_np = img_container.converted_polar_image.astype(np.float32)

            if self.use_cuda:
                return self._infer_with_iobinding(input_name, img_np)
            else:
                return self.sess.run(None, {input_name: img_np})
        except rt.capi.onnxruntime_pybind11_state.RuntimeException as e:
            error_message = str(e)
            if "Failed to allocate memory" in error_message or "BFCArena::AllocateRawInternal" in error_message:
                logging.error("GPU memory allocation failed. Consider using CPU execution with the option FORCE_CPU = True")
                raise MemoryError("GPU memory exhausted during inference.") from e
            else:
                raise

    def _infer_with_iobinding(self, input_name: str, img_np: np.ndarray):
        """Run inference using IOBinding so ONNX can use pinned host memory for the
        CPU->GPU transfer instead of allocating a new device buffer on every call."""
        io_binding = self.sess.io_binding()
        input_ort = rt.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(img_np), device_type='cuda', device_id=self.device_id
        )
        io_binding.bind_ortvalue_input(input_name, input_ort)
        # Outputs returned on CPU so postprocessing code needs no changes
        for out in self.sess.get_outputs():
            io_binding.bind_output(out.name, device_type='cpu')
        self.sess.run_with_iobinding(io_binding)
        return [io_binding.get_outputs()[i].numpy() for i in range(len(self.sess.get_outputs()))]