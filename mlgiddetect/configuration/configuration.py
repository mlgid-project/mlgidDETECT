import os
import sys
import logging
import yaml
import cv2

class Config:
    def __init__(self, config_file = None, args = None):
        self.init_default()
        if config_file is not None:
            self.config_file = config_file
            self.load_config()
        if args is not None:
            self.load_args(args)
        self.check_cuda_support()
        self.set_logging_level()
    
    def init_default(self):
        self.GENERAL_DEBUG = False
        #If True, the model is redownloaded each run
        self.MODEL_REDOWNLOAD = False
        #Model type used for inference, can be either "dino" or "faster_rcnn"
        self.MODEL_TYPE = 'dino'
        #If True, force model inference to be performed using the CPU 
        self.MODEL_FORCE_CPU = False
        #if True: activate detection-level ensemble (dino only): fuse ONNX_BASE + ONNX_ENSEMBLE.
        self.MODEL_ENSEMBLE_ENABLED = False
        #'base' -> the model mlgidDETECT downloads automatically for MODEL_TYPE; or a path to an .onnx file.
        self.MODEL_ONNX_BASE = 'base'
        #second model, used only when MODEL_ENSEMBLE_ENABLED is True (dino). This can either be 'ssl_pretrain', 'base', or any onnx file path
        self.MODEL_ONNX_ENSEMBLE = None
        #Input paths to dataset (pygid NeXus h5) or image (e.g. tif)
        self.INPUT_IMGPATH = None
        self.INPUT_DATASET = None
        #Used to evaluate model performance. If True data/analysis/frameXXXXX/fitted_peaks in *.h5 will be treated as GT boxes to calculate APr (only works for datasets not images).  
        self.INPUT_LABELED = False
        #Scale of the reciprocal-space image in pixels per Å⁻¹; overwritten by the loader from the actual data.
        self.GEO_PIXELPERANGSTROEM = 500
        #Pixel dimensions [q_z, q_xy] of the reciprocal-space image; overwritten with the real array shape on load.
        self.GEO_RECIPROCAL_SHAPE = [1501,1501]
        #Maximum q value (image-corner diagonal, sqrt(q_z² + q_xy²)) used to convert polar detections back to q; None means it is derived at runtime.
        self.GEO_QMAX = None
        #use CUDA for preprocessing if available
        self.PREPROCESSING_CUDA = False
        #use quazipolar transformation for preprocessing 
        self.PREPROCESSING_QUAZIPOLAR = False
        #flip the image horizontally for preprocessing
        self.PREPROCESSING_FLIPHORIZONTAL = False
        #use to perform polar conversion 
        self.PREPROCESSING_POLAR_CONVERSION = True
        #model input size
        self.PREPROCESSING_POLAR_SHAPE = [512,1024]
        #perform clipping of pixel values for preprocessing
        self.PREPROCESSING_PERFORMCLIPPING = True
        #use log contrast correction for preprocessing
        self.PREPROCESSING_LOG = True
        #model trained on histeq input
        self.PREPROCESSING_HISTOGRAMEQUALIZATION = True
        #upper clipping percentile for preprocessing
        self.PREPROCESSING_HIGHERCLIPPINGPERCENTILE = 99.5
        #lower clipping percentile for preprocessing
        self.PREPROCESSING_LOWERCLIPPINGPERCENTILE = 5.0
        #image output folder
        self.OUTPUT_FOLDER = './outputs/'
        #image prefix for saving images
        self.OUTPUT_IMAGEPREFIX = ''
        #minimum score for peaks (eval_on_dataset overrides to 0.1)
        self.POSTPROCESSING_SCORE = 0.4
        #NMS for non-class-aware models (legacy single-class model uses this threshold)
        self.POSTPROCESSING_NMSIOU = 0.4
        #class-aware NMS for the 2-class ring/segment dino model (segment=0, ring=1).
        #Leave False for the legacy single-class (91-class) model. Mirror of mlgidDETECT_DINO.
        self.POSTPROCESSING_CLASSAWARE_NMS = False
        self.POSTPROCESSING_NMSIOU_RING = 0.1
        self.POSTPROCESSING_NMSIOU_SEG = 0.4


    def load_config(self):
        if not os.path.isfile(self.config_file):
            logging.error(f"Configuration file not found: {self.config_file}")
            sys.exit()

        try:        
            with open(self.config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logging.debug(f"Configuration successfully loaded from: {self.config_file}")
        except yaml.YAMLError as e:
            logging.error(f"YAML parsing error in config file: {e}")
            sys.exit()
        except Exception as e:
            logging.exception(f"Unexpected error while loading config: {e}")
            sys.exit()

        # Set attributes dynamically
        for section, settings in config.items():
            for key, value in settings.items():
                setattr(self, f"{section}_{key}", value)


    def load_args(self, args):
        if args.epoch:
            self.EVAL_EPOCH = args.epoch
        if args.output_folder:
            self.EVAL_OUTPUT_FOLDER = args.output_folder
        if args.input_dataset:
            self.INPUT_DATASET = args.input_dataset
        if args.image_path:
            self.INPUT_IMGPATH = args.image_path


    def check_cuda_support(self):
        if self.PREPROCESSING_CUDA and (cv2.cuda.getCudaEnabledDeviceCount() > 0):
            try:
                import cupy as cp
                logging.info("Using CUDA for preprocessing!")
                self.PREPROCESSING_CUDA = True
            except ImportError:
                self.PREPROCESSING_CUDA = False
                logging.info("Cupy not installed, fallback to CPU!")
                logging.info("CUDA support for preprocessing not available. Use the script 'setup_cuda.py' to install it.\n The inference might still run on the GPU though.")
        elif self.PREPROCESSING_CUDA:
            self.PREPROCESSING_CUDA = False
            logging.info("CUDA support for preprocessing not available. Use the script 'setup_cuda.py' to install it.\n The inference might still run on the GPU though.")
        else:
            self.PREPROCESSING_CUDA = False

    def set_logging_level(self):
        if self.GENERAL_DEBUG:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
