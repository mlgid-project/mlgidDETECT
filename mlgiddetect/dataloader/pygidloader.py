"""EXAMPLE USAGE:
    dataset = PyGIDDataset(config, config.INPUT_DATASET, preprocess_func=standard_preprocessing, buffer_size=10)
    for i, img_container in enumerate(dataset):
        giwaxs_img = img_container.converted_polar_image
        boxes, scores, reciprocal_boxes = imp.infer(giwaxs_img, config = img_container.config)
        img_container.scores = scores
        img_container.boxes = boxes
        dataset.export_pygid(img_container)
    dataset.close()"""
import sys
import math
import logging
import numpy as np
import cv2 as cv
from dataclasses import dataclass, field
from typing import Tuple
from multiprocessing import Process, Queue, Lock
from h5py import File, Group
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
    
    def reciprocal_peaks_to_polar_boxes(self, img: ImageContainer) -> None:
        """Calculates the coordinates of the boxes in the polar coordinates

        Args:
            img (GIWAXSImage): ImageObject of GIWAXS image
        """
        polar_shape = self.polar_img_shape
        reciprocal_labels = img.reciprocal_labels
        polar_labels = img.polar_labels

        img.radius = img.radius/math.sqrt(img.q_xy**2+img.q_z**2)
        img.radius_width = img.radius_width/math.sqrt(img.q_xy**2+img.q_z**2)

        r_scale = polar_shape[1] 
        a_scale = polar_shape[0] / 90

        radii = img.radius * r_scale
        widths = img.radius_width  * r_scale / 2
        angles = img.angle * a_scale
        angles_std = img.angle_width * a_scale / 2

        boxes = np.stack([
            radii - widths, angles - angles_std, radii + widths, angles + angles_std
        ], -1)

        if  self.config.PREPROCESSING_QUAZIPOLAR:
            rs = ((boxes[:, 0] + boxes[:, 2]) / 2) / polar_shape[1]

            coef = 0.6 / (1e-4 + rs)

            boxes[:, 1::2] /= coef[:, None]
        
        polar_labels.boxes = boxes
        polar_labels.radii = radii
        polar_labels.widths = widths
        polar_labels.angles = angles
        polar_labels.angles_std = angles_std
        polar_labels.confidences = reciprocal_labels.confidences
        polar_labels.intensities = reciprocal_labels.intensities
        polar_labels.img_nr = reciprocal_labels.img_nr
        polar_labels.img_name = reciprocal_labels.img_name

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
            img_container.polar_img_shape = data_loader.polar_img_shape
            data_loader.file_locked.acquire()
            f = File(data_loader.path, 'r')
            group = f[key]
            img_container.raw_reciprocal = np.nan_to_num(group['data/img_gid_q'][i])
            img_container.h5_group = group.name
            img_container.q_z = group['data/q_z'][-1]
            img_container.q_xy = group['data/q_xy'][-1]
            if key + '/data/analysis/' + 'frame' + str(i).zfill(5) + '/fitted_peaks/' in f:
                fill_from_fitted_peaks(img_container, group['data/analysis/' + 'frame' + str(i).zfill(5) + '/fitted_peaks'])
                data_loader.reciprocal_peaks_to_polar_boxes(img_container)        
            f.close()
            data_loader.file_locked.release()               
            img_container.nr = i
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

def fill_from_fitted_peaks(img_container, results_array):
    img_container.angle = results_array['angle']
    img_container.angle_width = results_array['angle_width']
    img_container.radius = results_array['radius']
    img_container.radius_width = results_array['radius_width']
    img_container.qzqxyboxes = [results_array['q_z'],results_array['q_xy']]
    img_container.is_ring = results_array['is_ring']
    img_container.scores = results_array['score']
    img_container.reciprocal_labels.confidences = results_array['visibility']
    return img_container

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
            group = f.require_group(source_path)
            if source_path + '/detected_peaks'  in f:
                del group['detected_peaks']

            results_array = get_results_array(img_container)
            group.create_dataset('detected_peaks', data=results_array, dtype=pygid_results_dtype)

        except:
            f.close()
            data_loader.file_locked.release()           
            continue
        f.close()
        data_loader.file_locked.release()