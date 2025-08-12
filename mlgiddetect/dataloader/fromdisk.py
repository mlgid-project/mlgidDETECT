import tifffile
import numpy as np
import logging
from mlgiddetect.dataloader import ImageContainer
logging.getLogger('PIL').setLevel(logging.INFO)

def load_img_from_disk(config):
    img_container = ImageContainer()    
    img_container.raw_reciprocal = np.nan_to_num(tifffile.imread(config.INPUT_IMGPATH))
    config.GEO_RECIPROCAL_SHAPE = list(img_container.raw_reciprocal.shape)
    img_container.config = config
    logging.info('Loaded image sucessfully from ' + config.INPUT_IMGPATH)    
    return img_container