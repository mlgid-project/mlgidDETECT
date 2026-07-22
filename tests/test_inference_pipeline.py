# End-to-end tests for the code path downstream packages (mlgidBASE) drive:
# Config() -> Inference(config) -> standard_preprocessing -> infer -> standard_postprocessing.
#
# mlgidBASE's CI segfaulted (exit 139) because Inference used to request the CUDA
# execution provider whenever the onnxruntime-gpu wheel was installed, even on
# machines without a GPU/driver. These tests run the real provider selection and a
# real model on every supported Python (and therefore every pinned onnxruntime-gpu
# version), so a regression fails here before it ships to PyPI.

import os
import pytest

from mlgiddetect.configuration import Config
from mlgiddetect.inference import Inference
from mlgiddetect.inference import inference as inference_module
from mlgiddetect.preprocessing import standard_preprocessing
from mlgiddetect.postprocessing import standard_postprocessing
from mlgiddetect.dataloader import load_img_from_disk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_IMG = os.path.join(REPO_ROOT, 'inputs', 'w4_mapbbr32.tif')


def test_cuda_probe_does_not_crash():
    # Must return a plain bool on any machine, with or without a GPU/driver.
    assert inference_module._cuda_driver_usable() in (True, False)


def test_no_cuda_driver_selects_cpu(monkeypatch):
    # The mlgidBASE CI scenario: onnxruntime-gpu wheel installed (so
    # get_available_providers() lists CUDAExecutionProvider) but no usable
    # driver. Session creation must land on the CPU provider, not crash.
    monkeypatch.setattr(inference_module, '_cuda_driver_usable', lambda: False)
    sess = Inference(Config()).sess
    assert sess.get_providers() == ['CPUExecutionProvider']


def test_force_cpu_wins_over_gpu():
    config = Config()
    config.MODEL_FORCE_CPU = True
    sess = Inference(config).sess
    assert sess.get_providers() == ['CPUExecutionProvider']


def test_provider_selection_matches_environment():
    # With a real driver the CUDA provider must be first (with CPU fallback);
    # without one (e.g. GitHub runners) only CPU may be requested.
    import onnxruntime as rt
    sess = Inference(Config()).sess
    cuda_possible = ('CUDAExecutionProvider' in rt.get_available_providers()
                     and inference_module._cuda_driver_usable())
    if cuda_possible:
        assert sess.get_providers()[0] == 'CUDAExecutionProvider'
        assert 'CPUExecutionProvider' in sess.get_providers()
    else:
        assert sess.get_providers() == ['CPUExecutionProvider']


@pytest.mark.parametrize('model_type', ['dino', 'faster_rcnn'])
def test_full_detection_pipeline(model_type):
    # Mirrors mlgidBASE's _detect_test_dino/_detect_test_faster: default config,
    # auto-downloaded model, full preprocess -> infer -> postprocess.
    config = Config()
    config.MODEL_TYPE = model_type
    config.INPUT_IMGPATH = EXAMPLE_IMG

    img_container = load_img_from_disk(config)
    (img_container.converted_polar_image,
     img_container.raw_polar_image,
     img_container.converted_mask) = standard_preprocessing(config, img_container.raw_reciprocal)

    imp = Inference(config)
    raw_results = imp.infer(img_container)
    img_container = standard_postprocessing(img_container, raw_results)

    assert len(img_container.boxes) == len(img_container.scores)
    assert len(img_container.boxes) > 0, f'{model_type} found no peaks in the example image'
