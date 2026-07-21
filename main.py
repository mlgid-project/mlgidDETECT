import argparse
import logging
import numpy as np
import multiprocessing as mp
mp.set_start_method('spawn',force=True)

from mlgiddetect.configuration import Config
from mlgiddetect.inference import load_sessions
from mlgiddetect.preprocessing import standard_preprocessing
from mlgiddetect.postprocessing import standard_postprocessing
from mlgiddetect.postprocessing.postprocessing import ensemble_postprocessing
from mlgiddetect.dataloader import load_img_from_disk, PyGIDDataset, H5GIWAXSDataset
from mlgiddetect.evaluation import eval_on_dataset
from mlgiddetect.export import plot_img_with_boxes, export_pygid_h5

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    # Argparser, makes the script usable form terminal
    parser = argparse.ArgumentParser(description="Awesome GIWAXS script")
    parser.add_argument("--config_file", help="Path to config file")
    parser.add_argument("--epoch", help="epoch, for training purposes")
    parser.add_argument("--output_folder")
    parser.add_argument("--onnx_path")
    parser.add_argument("--input_dataset")
    parser.add_argument("--image_path")
    args = parser.parse_args()

    if args.config_file:
        config = Config(args.config_file, args)
    else:
        config = Config('/home/testuser/mlgidDETECT/dino.yaml', args)
    if config.INPUT_DATASET:
        #evaluation on labeled dataset
        if config.INPUT_LABELED:
            eval_on_dataset(config, standard_preprocessing)
        else:
            #add detected boxes to PyGIDDataset dataset  
            dataset = PyGIDDataset(config, config.INPUT_DATASET, preprocess_func=standard_preprocessing, buffer_size=10)
            sessions = load_sessions(config)
            for i, img_container in enumerate(dataset):
                logging.info("Processing image %s", i)
                if len(sessions) > 1:
                    config.POSTPROCESSING_CLASSAWARE_NMS = True
                    img_container = ensemble_postprocessing(img_container, [s.infer(img_container) for s in sessions])
                else:
                    img_container = standard_postprocessing(img_container, sessions[0].infer(img_container))
                dataset.export_pygid(img_container)
            dataset.close()

    elif config.INPUT_IMGPATH:
        #peak detection on single image
        sessions = load_sessions(config)
        img_container = load_img_from_disk(config)
        img_container.converted_polar_image, img_container.raw_polar_image, img_container.converted_mask = standard_preprocessing(config, img_container.raw_reciprocal)
        if len(sessions) > 1:
            config.POSTPROCESSING_CLASSAWARE_NMS = True
            img_container = ensemble_postprocessing(img_container, [s.infer(img_container) for s in sessions])
        else:
            raw_results = sessions[0].infer(img_container)
            img_container = standard_postprocessing(img_container, raw_results)
        plot_img_with_boxes(config, np.transpose(img_container.converted_polar_image[0], (1,2,0)), img_container.scores, img_container.boxes, config.OUTPUT_FOLDER, name='testoutput')
        export_pygid_h5(config, img_container=img_container)