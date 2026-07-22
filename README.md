# mlgidDETECT
[![Python version](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12%7C3.13%7C3.14-blue.svg)](https://www.python.org/)

This package is included in the [`mlgidBASE` package](https://github.com/mlgid-project/mlgidBASE) and can be used as part of the `mlgid` pipeline. 

## Clone repository
* Clone with ssh (recommended)
```git clone git@github.com:mlgid-project/mlgidDETECT.git```
* If it fails, use https:
```git clone https://github.com/mlgid-project/mlgidDETECT.git```


## Installation

### Install Conda environment (recommended)
* Install miniconda
[https://docs.anaconda.com/miniconda/#quick-command-line-install](https://docs.anaconda.com/miniconda/#quick-command-line-install)

* Move into directory:
```cd mlgidDETECT```

* (Option 1) Create environment with CPU and optional GPU inference\
```cd setup```\
```conda env create -f conda_cpu.yaml```\
```conda activate mlgiddetect-cpu```\

* (Option 2) Create environment with with additional GPU preprocessing\
```cd setup```\
```python setup_cuda.py```\
```conda activate mlgiddetect-gpu```\
```conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}```\
```conda deactivate```\
```conda activate mlgiddetect-gpu```\
Set ```PREPROCESSING CUDA: True``` in the config file


### Install package with pip
* Install package \
```pip install mlgiddetect```

## Usage

### With a PyGIDDataset

```python main.py --input_dataset=/home/testuser/dataset.h5```

### With a single image

```python main.py --image_path=./w4_mapbbr32.tif```

### With a config file

```python main.py --config_file=./faster_rcnn.yaml```

### Model selection & ensemble

Models are configured in the `MODEL` section of a config file. Each model slot accepts either a
**keyword** (a model that is auto-downloaded and cached in `~/.local/share/mlgiddetect/`) or a
**path** to a local `.onnx` file:

| keyword | model |
|---|---|
| `base` | default model for `TYPE` (the 2-class class-aware dino model, or the faster_rcnn model) |
| `ssl_pretrain` | SSL-pretrained 2-class dino model |
| `dino_old` | legacy single-class (91-class) dino model |

```yaml
MODEL:
  TYPE: 'dino'                 # 'dino' or 'faster_rcnn'
  ONNX_BASE: base              # model used on its own; a keyword or a path to an .onnx
  ENSEMBLE_ENABLED: False      # dino only: fuse ONNX_BASE + ONNX_ENSEMBLE (detection-level)
  ONNX_ENSEMBLE: ssl_pretrain  # second model, used only when ENSEMBLE_ENABLED is True
```

- **Single model** — `ENSEMBLE_ENABLED: False` runs `ONNX_BASE` alone.
- **Ensemble (dino only)** — `ENSEMBLE_ENABLED: True` runs `ONNX_BASE` + `ONNX_ENSEMBLE` and fuses
  their detections with class-aware NMS. Both members must be 2-class ring/segment models, so keep
  `POSTPROCESSING.CLASSAWARE_NMS: True`.
- **faster_rcnn** — the ensemble is ignored; the single `ONNX_BASE` model is always used.

To use the legacy model, set `ONNX_BASE: dino_old`, `ENSEMBLE_ENABLED: False` and
`POSTPROCESSING.CLASSAWARE_NMS: False` (the 91-class model needs single-class NMS).

### Using the PyPI package

Use [mlgidDETECT_tutorial.ipynb](https://github.com/mlgid-project/mlgidDETECT/blob/main/mlgidDETECT_tutorial.ipynb) to get started.


### GPU support
The pip package depends on the GPU build of ONNX Runtime, pinned per
Python version: `onnxruntime-gpu==1.26.0` on Python 3.11+, `1.23.2` on
3.10 and `1.19.2` on 3.9 (the last releases with wheels for those
Pythons). All three are CUDA 12 builds (the same CUDA generation the
conda GPU environment ships); newer `onnxruntime-gpu` wheels (1.27+)
require the CUDA 13 runtime and load it at import time, which breaks
environments without it. If the GPU build is installed and CUDA is
available, it is automatically used for inference.

For CPU-only machines, replace it with the CPU build (never install
both at the same time, they share the same `onnxruntime` module and
overwrite each other):

```
pip uninstall -y onnxruntime-gpu
pip install onnxruntime
```

To use CUDA for preprocessing, use the install instructions for GPU support.
