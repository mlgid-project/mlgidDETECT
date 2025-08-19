import logging
from mlgiddetect.dataloader import H5GIWAXSDataset
from mlgiddetect.evaluation import Evaluator, get_full_conf_results
from mlgiddetect.export import plot_img_with_boxes, write_logs, write_single_log
from mlgiddetect.utils import open_pkl_file
from mlgiddetect.postprocessing import SmallQFilter, standard_postprocessing
import numpy as np
import pickle
import torch
from torch import Tensor
from torchvision.ops import nms
from torchvision.utils import save_image
from mlgiddetect.utils import open_pkl_file
from matplotlib import pyplot as plt

postprocessing =  SmallQFilter(50)

def filter_non_elong(pred_boxes):
    y_extent = pred_boxes[:,3] - pred_boxes[:,1]
    x_extent = pred_boxes[:,2] - pred_boxes[:,0]
    keep = x_extent*1.15 < y_extent
    return keep

def eval_on_dataset(config, prepro_func, img_processing, postpro_func=standard_postprocessing, dataset = None, export_path = None):
    if dataset is None:
        if config.INPUT_DATASET.endswith(('pkl','pickle','p')):
            dataset = open_pkl_file(config.INPUT_DATASET)
        else:
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

    logging.info('Started evaluation')
    for i, img_container in enumerate(dataset):
        giwaxs_img = img_container.converted_polar_image
        labels = img_container.polar_labels
        confidences = img_container.polar_labels.confidences
        gt_boxes = Tensor(labels.boxes)

        if postpro_func:
            img_container = standard_postprocessing(img_container, img_processing.infer(img_container))
        else:
            img_container = img_processing.infer(img_container)
        pred_boxes = img_container.boxes
        scores = Tensor(img_container.scores)
            #plot_img_with_boxes(config, np.transpose(giwaxs_img[0], (1,2,0)), scores, pred_boxes, config.OUTPUT_FOLDER, i)
            #plot_imgs([np.transpose(giwaxs_img[0], (1,2,0))],config.OUTPUT_FOLDER, i)

        if export_path is not None:
            results['images'].append(Tensor(giwaxs_img[0]).cpu())
            results['raw_images'].append(Tensor(img_container.raw_polar_image))
            results['gt_boxes'].append(gt_boxes)
            results['gt_scores'].append(Tensor(confidences).cpu())
            results['pred_boxes'].append(pred_boxes)
            results['pred_scores'].append(scores)


        idx_keep = nms(pred_boxes, scores, 0.09)
        pred_boxes = pred_boxes[idx_keep]
        scores = scores[idx_keep]

        pred_boxes, scores = postprocessing(pred_boxes, scores)
        idx_elong = filter_non_elong(pred_boxes)
        scores = scores[idx_elong]
        pred_boxes = pred_boxes[idx_elong]

        logging.info('evaluating img nr ' + str(i))
        evaluator.get_exp_metrics(pred_boxes, scores, gt_boxes, confidences)

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
    
def eval_from_pkl(pkl_file: str, score_thres: float = 0.01, iou_threshold : float = .5):
    detection_results = open_pkl_file(pkl_file)
    evaluator = Evaluator()

    for i, _ in enumerate(detection_results['images']):
        scores = detection_results['pred_scores'][i]
        pred_boxes = detection_results['pred_boxes'][i][scores>score_thres]        
        gt_boxes = detection_results['gt_boxes'][i]#.numpy()
        gt_scores = detection_results['gt_scores'][i].numpy()
        #scores = detection_results['pred_scores'][i][scores>score_thres]        
        scores = scores[scores>score_thres]
        indices = nms(pred_boxes, scores, iou_threshold)
        pred_boxes = pred_boxes[indices]#.numpy()
        scores = scores[indices].numpy()
        evaluator.get_exp_metrics(pred_boxes, scores, gt_boxes, gt_scores)

    df1, df2 = get_full_conf_results(evaluator.metrics)
    print('------evaluation-results-----')
    print(df1)
    print(df2)