import os
from pathlib import Path
import logging
import urllib.request
import pickle
import ssl
import time
import sys
import appdirs

MODEL_URLS = {
    'dino': 'https://huggingface.co/mlgid-project/mlgidDETECTdino/resolve/main/model.onnx?download=true',
    'frcnn': 'https://huggingface.co/mlgid-project/mlgidDETECTfrcnn/resolve/main/model.onnx?download=true'
}

logger = logging.getLogger()


def check_filepath(config, filepath: Path):
    """
    Checks if a given file path is valid, if the file exists, and if the file extension is .onnx.

    Args:
        filepath (str): The file path to check.

    Returns:
        bool: True if the filepath is valid and the file extension is .onnx, False otherwise.
    """

    try:
        # Check if the file exists
        if not os.path.exists(filepath):
            logger.info("Model file does not exist.")
            return False

        # Check if the path is a file and not a directory
        if not os.path.isfile(filepath):
            logger.error("Model path is not a file.")
            return False

        # Check if the file extension is .onnx
        _, ext = os.path.splitext(filepath)
        if ext.lower() != '.onnx':
            logger.error("Provided model file is not ONNX.")
            return False
        return True

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False
    
def check_onnx_filepath(config, onnx_path: Path):
    if check_filepath(config, onnx_path):
        return True
    else:
        return False
    
def get_data_dir(appname: str = 'mlgiddetect'):
    return Path(appdirs.user_data_dir(appname))

def download(config, model_name: str = None,  source: str = None, destination: str = None) -> Path:
    """
    Downloads a file from a source URL to a destination.

    Args:
        source (str): The source URL. If not provided, a default URL is used.
        destination (str): The path where the file should be saved.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        if model_name is None:
            model_name = 'frcnn'

        source = MODEL_URLS[model_name]
        if destination is None:
            destination = get_data_dir() / 'model_name.onnx'
        #create download dir if it does not exist
        get_data_dir().mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting download of model file to {destination}")
        ssl._create_default_https_context = ssl._create_unverified_context

        _download_start_time = [time.time()]
        _last_time = [time.time()]
        _last_count = [0]

        def _progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                downloaded = min(downloaded, total_size)

            now = time.time()
            elapsed = now - _last_time[0]
            if elapsed >= 0.5:
                bytes_since = downloaded - _last_count[0]
                speed = bytes_since / elapsed  # bytes/s
                _last_time[0] = now
                _last_count[0] = downloaded

                percent = downloaded / total_size * 100
                done = int(percent / 2)
                bar = '#' * done + '-' * (50 - done)
                total_mb = total_size / 1e6
                dl_mb = downloaded / 1e6
                speed_str = f"{speed/1e6:.2f} MB/s" if speed >= 1e5 else f"{speed/1e3:.1f} KB/s"
                sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  {dl_mb:.1f}/{total_mb:.1f} MB  {speed_str}   ")
                sys.stdout.flush()

        urllib.request.urlretrieve(source, destination, reporthook=_progress_hook)
        total_size = os.path.getsize(destination)
        total_mb = total_size / 1e6
        elapsed = time.time() - _download_start_time[0]
        avg_speed = total_size / elapsed if elapsed > 0 else 0
        speed_str = f"{avg_speed/1e6:.2f} MB/s" if avg_speed >= 1e5 else f"{avg_speed/1e3:.1f} KB/s"
        sys.stdout.write(f"\r  [{'#'*50}] 100.0%  {total_mb:.1f}/{total_mb:.1f} MB  {speed_str}   \n")
        sys.stdout.flush()
        logging.info(f"Model file downloaded successfully from {source} to {destination}")
        return destination
    
    except Exception as e:
        logging.error(f"Failed to download file. Source: {source}, Destination: {destination}, Error: {str(e)}")
        return None

def get_model_path(config, model_name: str = 'faster_rcnn.onnx') -> Path:
    
    if config.MODEL_TYPE == 'dino':
        model_name = 'dino'
    if config.MODEL_TYPE == 'faster_rcnn':
        model_name = 'frcnn'
    data_dir = get_data_dir()
    onnx_dir = data_dir / (model_name + '.onnx')
    if not check_onnx_filepath(config, onnx_dir) or config.MODEL_REDOWNLOAD:
        return download(config, model_name, destination=onnx_dir)
    return str(onnx_dir)

def open_pkl_file(file_path: str):
    with open(file_path, 'rb') as handle:
        dataset = pickle.load(handle)
    return dataset