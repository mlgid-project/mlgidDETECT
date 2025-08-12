import sys
import logging
import numpy as np
import onnxruntime as rt
from mlgiddetect.utils import path_utils
from mlgiddetect.dataloader import ImageContainer

class Inference:
    def __init__(self, config):
        self.config = config
        model_path = path_utils.get_model_path(config)
        if model_path is None:
            logging.error('could not get model file. Exiting')
            sys.exit()
        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 4
        logging.info("Loading model")
        available_providers = rt.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

        if preferred_providers[0] == 'CUDAExecutionProvider':
            logging.info("Using the GPU for inference")

        self.sess = rt.InferenceSession(model_path, providers=preferred_providers, sess_options=sess_options)

    def infer(self, img_container: ImageContainer):
        # Run inference with ONNX Runtime
        input_name = self.sess.get_inputs()[0].name
        return self.sess.run(None, {input_name: img_container.converted_polar_image.astype(np.float32)})