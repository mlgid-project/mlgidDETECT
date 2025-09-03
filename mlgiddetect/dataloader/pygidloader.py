"""EXAMPLE USAGE:
    dataset = PyGIDDataset(config, config.INPUT_DATASET, preprocess_func=standard_preprocessing, buffer_size=10)
    for i, img_container in enumerate(dataset):
        giwaxs_img = img_container.converted_polar_image
        boxes, scores, reciprocal_boxes = imp.infer(giwaxs_img, config = img_container.config)
        img_container.scores = scores
        img_container.boxes = boxes
        dataset.export_pygid(img_container)
    dataset.close()"""
from dataclasses import dataclass, field
import sys
from typing import Tuple
from multiprocessing import Process, Queue, Value, Lock
from typing import Iterator
import logging
import numpy as np
import cv2 as cv
from h5py import File, Group, Dataset
from mlgiddetect.preprocessing import contrast_correction
from mlgiddetect.dataloader import ImageContainer


sys.path.append("..")

DEFAULT_CLAHE_LIMIT: float = 2000.
DEFAULT_CLAHE_COEF: float = 500.
DEFAULT_POLAR_SHAPE: Tuple[int, int] = (512, 512)
MAX_GRID_CACHE_SIZE: int = 5
DEFAULT_BEAM_CENTER: Tuple[float, float] = (0, 0)
DEFAULT_ALGORITHM: int = cv.INTER_CUBIC

@dataclass
class PyGIDDataset():
    """Container for a GIWAXS dataset. Contains multiple images with Labels.
    the images are loaded in a serarate process and stored in image_queue.
    The dataset can be iterated over with the function iter_images:
    e.g.

    data = H5GIWAXSDataset(dataset, buffer_size=5, unskewed_polar=True)
    for i, giwaxs_img_container in enumerate(data.iter_images()):

        giwaxs_img = giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = giwaxs_img_container.raw_polar_image
        labels = giwaxs_img_container.polar_labels
        fits = giwaxs_img_container.fits
        img_container.scores =
        img_container.boxes = 
    """

    # Path to the dataset
    config: 'Config' = None
    path: str = None
    preprocess_func: callable = None
    file_group: Group = None
    #min confidence of loaded boxes
    min_confidence: float = None
    #shape of the desired polar images
    polar_img_shape: tuple = DEFAULT_POLAR_SHAPE
    load_worker: Process = None
    write_worker: Process = None
    #buffer size of the worker. E.g. 3 are loaded in advance by default.
    buffer_size: int = 5
    image_metrics: list = field(default_factory=list)
    file_locked = Lock()


    def __post_init__(self):
        if self.config is not None:
            self.polar_img_shape: tuple = self.config.PREPROCESSING_POLAR_SHAPE
            self.path: str = self.config.INPUT_DATASET

        self.file_locked = Lock()
        self.writer_running = Lock()
        self.read_queue = Queue(self.buffer_size)
        self.write_queue = Queue(self.buffer_size)
        Process(target=load_worker, args=[self], daemon=True).start()
        Process(target=write_worker, args=[self], daemon=False).start()
    
    def __iter__(self):
        return self

    def __next__(self):
        queue_object = self.read_queue.get()
        if queue_object is not None:
            return queue_object
        else:
            raise StopIteration

    def export_pygid(self, img_container: ImageContainer):
        self.write_queue.put(img_container)

    def close(self):
        self.write_queue.put(None)
        self.writer_running.acquire()


def load_worker(data_loader: PyGIDDataset):
    """worker for the interaction with the H5 file and a automatic conversion to polar coordinates
        Intended to be spawned as a separate process and enqueue the results

    Args:
        data_loader (H5GIWAXSDataset):
    """

    data_loader.file_locked.acquire()
    try:
        f = File(data_loader.path, 'r')
    except FileNotFoundError as e:
        logging.error('Could not find the dataset file: \'' + str(data_loader.path) + '\' Please check the path.')
        data_loader.read_queue.put(None)
        return
    file_keys = [key for key in f.keys()]
    data_loader.file_locked.release()

    for counter, key in enumerate(file_keys):
        data_loader.file_locked.acquire()
        f = File(data_loader.path, 'r')
        try:
            img_nrs = range(len(f[key]['data/img_gid_q'][()]))
            data_loader.config.GEO_QMAX = np.sqrt(((f[key]['data/q_z'][-1]) ** 2) + ((f[key]['data/q_xy'][-1]) ** 2))
            data_loader.config.GEO_PIXELPERANGSTROEM = f[key]['data/img_gid_q'][0].shape[0] / f[key]['data/q_z'][-1]

        except:
            f.close()
            data_loader.file_locked.release()
            continue

        f.close()
        data_loader.file_locked.release()

        for i in img_nrs:
            img_container = ImageContainer()
            img_container.config = data_loader.config
            data_loader.file_locked.acquire()
            f = File(data_loader.path, 'r')
            group = f[key]
            raw_img = group['data/img_gid_q'][i]
            img_container.h5_group = group.name
            img_container.q_z = group['data/q_z'][-1]
            img_container.q_xy = group['data/q_xy'][-1]
            f.close()
            data_loader.file_locked.release()               
            img_container.nr = i
            img_container.raw_reciprocal = np.nan_to_num(raw_img)
            img_container.converted_polar_image, img_container.raw_polar_image, img_container.converted_mask = data_loader.preprocess_func(data_loader.config, img_container.raw_reciprocal, counter)
            data_loader.read_queue.put(img_container)
    data_loader.read_queue.put(None)


pygid_results_dtype = np.dtype([
        ('amplitude', 'f4'),
        ('angle', 'f4'),
        ('angle_width', 'f4'),
        ('radius', 'f4'),
        ('radius_width', 'f4'),
        ('q_z', 'f4'),        
        ('q_xy', 'f4'),
        ('theta', 'f4'),
        ('score', 'f4'),
        ('A', 'f4'),
        ('B', 'f4'),
        ('C', 'i4'),
        ('is_ring', 'bool'),
        ('is_cut_qz', 'bool'),
        ('is_cut_qxy', 'bool'),
        ('visibility', 'i4'),
        ('id', 'i4'),
    ])

def get_results_array(img_container):
    results_array = np.zeros(len(img_container.radius_width), dtype=pygid_results_dtype)
    results_array['amplitude'] = [0] * len(img_container.radius_width)
    results_array['angle'] = img_container.angle
    results_array['angle_width'] = [abs(num) for num in img_container.angle_width]
    results_array['radius'] = img_container.radius
    results_array['radius_width'] = img_container.radius_width
    results_array['q_z'] = img_container.qzqxyboxes[0]    
    results_array['q_xy'] = img_container.qzqxyboxes[1]
    results_array['theta'] = [0] * len(img_container.radius_width)
    results_array['A'] = [0] * len(img_container.radius_width)
    results_array['B'] = [0] * len(img_container.radius_width)
    results_array['C'] = [0] * len(img_container.radius_width)
    results_array['is_ring'] = img_container.is_ring
    results_array['is_cut_qz'] = [0] * len(img_container.radius_width)
    results_array['is_cut_qxy'] = [0] * len(img_container.radius_width)
    results_array['visibility'] = [0] * len(img_container.radius_width)
    results_array['score'] = img_container.scores
    results_array['id'] = list(range(len(img_container.radius)))
    return results_array


def write_worker(data_loader: PyGIDDataset):
    """worker for storing the results in the H5 file
        Intended to be spawned as a separate process
    Args:
        data_loader (H5GIWAXSDataset):
    """

    data_loader.writer_running.acquire()
    while True:
        img_container = data_loader.write_queue.get()
        if img_container is None:
            logging.info("Finished! All results are stored in the H5 file")
            data_loader.writer_running.release()            
            return
        source_path = f'{img_container.h5_group}/data/analysis/frame' +  str(img_container.nr).zfill(5) +'/'
        data_loader.file_locked.acquire()
        f = File(data_loader.path, 'a')
        try:
            if source_path not in f:
                try:
                    group = f.create_group(source_path)
                except:
                    data_loader.file_locked.release()
                    continue
            else:
                group = f[source_path]
                for name in list(group.keys()):
                    if isinstance(group[name], Dataset):
                        del group[name]

            results_array = get_results_array(img_container)
            group.create_dataset('detected_peaks', data=results_array, dtype=pygid_results_dtype)

        except:
            f.close()
            data_loader.file_locked.release()           
            continue
        f.close()
        data_loader.file_locked.release()