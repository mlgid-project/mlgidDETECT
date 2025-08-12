from mlgiddetect.configuration import Config
import numpy as np
from h5py import File
from pathlib import Path

def export_to_h5(config: Config, img, reciprocal_boxes: np.array, scores: np.array):
    filename = 'sourceImages/' + Path(config.INPUT_IMGPATH).stem
    with File(config.OUTPUT_H5PATH, 'w') as hf:
        g2 = hf.create_group(filename)
        g2.create_dataset('image', data=img)
        g3 = hf.create_group(filename + '/roi_data')
        g3.create_dataset('radius', data=reciprocal_boxes[0])
        g3.create_dataset('angle', data=reciprocal_boxes[1])
        g3.create_dataset('angle_std', data=reciprocal_boxes[2])
        g3.create_dataset('confidence_level', data=[.1 for x in range(len(reciprocal_boxes[0]))])
        g3.create_dataset('type', data=[2 for x in range(len(reciprocal_boxes[0]))])
        ciffile = ["not_set" for x in range(len(reciprocal_boxes[0]))]
        g3.create_dataset('cif_file', data=ciffile)
        g3.create_dataset('width', data=reciprocal_boxes[3])