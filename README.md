# mlgidDETECT

## Clone repository
* Clone with ssh (recommended)
```git clone git@github.com:mlgid-project/mlgidDETECT.git```
* If it fails, use https:
```git clone https://github.com/mlgid-project/mlgidGUI/mlgidDETECT.git```


## Installation

### Install Conda environment (recommended)
* Install miniconda
[https://docs.anaconda.com/miniconda/#quick-command-line-install](https://docs.anaconda.com/miniconda/#quick-command-line-install)

* Move into directory:
``` cd mlgidDETECT```

* (Option 1) Create environment with CPU only\
```conda env create -f conda_cpu.yaml```\
```conda activate mlgiddetect-cpu```

* (Option 2) Create environment with GPU support\
```python setup_cuda.py```
```conda activate mlgiddetect-gpu```
```conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}```
Set ```PREPROCESSING CUDA: True``` in the config file


### Install with pip (not recommended)
* Move into directory:
``` cd mlgidDETECT```
* Install package, recommended with option -e to install in editable mode
```pip install -e ./```

## Usage

### With a PyGIDDataset

```python main.py --input_dataset=/home/testuser/dataset.h5```

### With a single image

```python main.py --image_path=./w4_mapbbr32.tif```

### With a config file

```python main.py --config_file=./faster_rcnn.yaml```

### GPU support with CUDA 12.X
If a GPU is available, it is automatically used for inference.
To use CUDA for preprocesing, use the install instructions for GPU support.