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

### Using the PyPI package

Use [mlgidDETECT_tutorial.ipynb](https://github.com/mlgid-project/mlgidDETECT/blob/main/mlgidDETECT_tutorial.ipynb) to get started.


### GPU support with CUDA 12.X
If a GPU is available, it is automatically used for inference.
To use CUDA for preprocessing, use the install instructions for GPU support.
