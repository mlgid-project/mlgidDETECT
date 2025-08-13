import numpy as np
from h5py import File
from pathlib import Path
from mlgiddetect.configuration import Config
from mlgiddetect.dataloader import pygid_results_dtype

def export_to_h5_deprecated(config: Config, img, reciprocal_boxes: np.array, scores: np.array):
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

def export_pygid_h5(config: Config, img_container):

    if img_container.config.INPUT_IMGPATH:
        filename = Path(img_container.config.OUTPUT_FOLDER) / Path(img_container.config.INPUT_IMGPATH).name
    else:
        filename = Path(img_container.config.OUTPUT_FOLDER) / "detection_results.h5"

    with File(filename, 'a') as f:
        source_path = f'{img_container.h5_group}/data/analysis/frame' +  str(img_container.nr).zfill(5) +'/'
        group = f.create_group(source_path)
        results_struct = np.zeros(len(img_container.radius_width), dtype=pygid_results_dtype)
        results_struct['amplitude'] = [0] * len(img_container.radius_width)
        results_struct['angle'] = img_container.angle
        results_struct['angle_width'] = [abs(num) for num in img_container.angle_width]
        results_struct['radius'] = img_container.radius
        results_struct['radius_width'] = img_container.radius_width
        results_struct['q_xy'] = img_container.qzqxyboxes[1]
        results_struct['q_z'] = img_container.qzqxyboxes[0]
        results_struct['theta'] = [0] * len(img_container.radius_width)
        results_struct['A'] = [0] * len(img_container.radius_width)
        results_struct['B'] = [0] * len(img_container.radius_width)
        results_struct['C'] = [0] * len(img_container.radius_width)
        results_struct['is_ring'] = [0] * len(img_container.radius_width)
        results_struct['is_cut_qz'] = [0] * len(img_container.radius_width)
        results_struct['is_cut_qxy'] = [0] * len(img_container.radius_width)
        results_struct['visibility'] = [0] * len(img_container.radius_width)
        results_struct['score'] = img_container.scores
        results_struct['id'] = list(range(len(img_container.radius)))
        group.create_dataset('detected_peaks', data=results_struct, dtype=pygid_results_dtype)