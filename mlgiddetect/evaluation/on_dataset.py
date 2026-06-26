import logging
from mlgiddetect.dataloader import H5GIWAXSDataset, PyGIDDataset, detect_dataset_type
from mlgiddetect.evaluation import Evaluator, get_full_conf_results
from mlgiddetect.export import write_logs, write_single_log
from mlgiddetect.utils import open_pkl_file
from mlgiddetect.postprocessing import SmallQFilter, standard_postprocessing, boxes_polar_to_reciprocal, boxes_reciprocal_q_to_xy, polar_to_cartesian
from mlgiddetect.postprocessing.utils import onnx_to_xyxy, filter_boxes
from mlgiddetect.inference.inference import load_sessions
from mlgiddetect.postprocessing.postprocessing import ensemble_postprocessing
import pickle
from torch import Tensor
from torchvision.ops import nms
from mlgiddetect.utils import open_pkl_file

postprocessing =  SmallQFilter(50)

def eval_on_dataset(config, prepro_func, postpro_func=standard_postprocessing, dataset = None, export_path = None):
    if dataset is None:
        if config.INPUT_DATASET.endswith(('pkl','pickle','p')):
            dataset = open_pkl_file(config.INPUT_DATASET)
        elif detect_dataset_type(config.INPUT_DATASET) == 'pygid':
            #pyGID/NeXus labeled file (e.g. organic_labeled.h5): img_gid_q + fitted_peaks GT
            dataset = PyGIDDataset(config, preprocess_func=prepro_func, buffer_size=5, load_labels=True)
        else:
            #roi_data labeled file (e.g. 41_test.h5)
            dataset = H5GIWAXSDataset(config, config.INPUT_DATASET, preprocess_func=prepro_func, buffer_size=5)

        #save dataset
        """ ds = list(dataset)
        with open('40_labeled_3channel.pkl', 'wb') as handle:
            pickle.dump(ds, handle, protocol=4) """
    evaluator = Evaluator()
    if export_path is not None:
        results = {
            'images': list(),
            'raw_images': list(),
            'masks': list(),
            'gt_boxes': list(),
            'gt_scores': list(),
            'pred_boxes': list(),
            'pred_scores': list()
        }
    
    # dino ensemble (MODEL.ENSEMBLE_ENABLED) -> [ONNX_BASE, ONNX_ENSEMBLE] fused;
    # otherwise a single model (ONNX_BASE; faster_rcnn always lands here).
    sessions = load_sessions(config)

    logging.info('Started evaluation')
    try:
        for i, img_container in enumerate(dataset):
            img_container.config.POSTPROCESSING_SCORE = 0.1
            giwaxs_img = img_container.converted_polar_image
            labels = img_container.polar_labels
            confidences = img_container.polar_labels.confidences
            gt_boxes = Tensor(labels.boxes)

            if postpro_func:
                if len(sessions) > 1:
                    config.POSTPROCESSING_CLASSAWARE_NMS = True
                    raw_results_list = [s.infer(img_container) for s in sessions]
                    img_container = ensemble_postprocessing(img_container, raw_results_list)
                else:
                    img_container = standard_postprocessing(img_container, sessions[0].infer(img_container))
            else:
                img_container = sessions[0].infer(img_container)
            pred_boxes = img_container.boxes
            scores = Tensor(img_container.scores)

            if export_path is not None:
                results['images'].append(Tensor(giwaxs_img[0]).cpu())
                results['raw_images'].append(Tensor(img_container.raw_polar_image))
                results['gt_boxes'].append(gt_boxes)
                results['gt_scores'].append(Tensor(confidences).cpu())
                results['pred_boxes'].append(pred_boxes)
                results['pred_scores'].append(scores)

            logging.info('evaluating img nr ' + str(i))
            evaluator.get_exp_metrics(pred_boxes, scores, gt_boxes, confidences)
    finally:
        #PyGIDDataset spawns a non-daemon write_worker that must be joined; H5GIWAXSDataset.close() is a no-op
        if hasattr(dataset, 'close'):
            dataset.close()

    if export_path is not None:
        with open(export_path + '/object_detection_results.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=4)
        
    df1, df2 = get_full_conf_results(evaluator.metrics)

    print('------evaluation------')
    print(df1)
    print(df2)

    #logging for training

    if hasattr(config,'EVAL_EPOCH'):
        if config.PREPROCESSING_SPLIT != 1:
            write_logs('split_img, epoch ' + str(config.EVAL_EPOCH) + df1.to_string(), config.EVAL_OUTPUT_FOLDER, config)
            print('split-metrics:' + df1.to_string())
        
        elif config.PREPROCESSING_QUAZIPOLAR:
            write_single_log(str(df1.recall_total[0]), config.EVAL_OUTPUT_FOLDER, config)
            write_logs('quazipolar_img, epoch ' + config.EVAL_EPOCH  + df1.to_string(), config.EVAL_OUTPUT_FOLDER, config)
            print('quazipolar-metrics:' + df1.to_string())
    
        else:
            write_single_log(str(df1.recall_total[0]), config.EVAL_OUTPUT_FOLDER, config)
            write_logs('full_img, epoch ' + config.EVAL_EPOCH  + df1.to_string(), config.EVAL_OUTPUT_FOLDER, config)
            write_logs('full_img, epoch ' + config.EVAL_EPOCH  + df2.to_string(), config.EVAL_OUTPUT_FOLDER, config)
            print('single-metrics:' + df1.to_string())
            
        return df2['ap_total'].values[0]
