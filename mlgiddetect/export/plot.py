import logging
import numpy as np
from matplotlib import pyplot as plt
from mlgiddetect.utils import open_pkl_file
from torchvision import utils
from torch import Tensor

logging.getLogger('matplotlib').setLevel(logging.WARNING)

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_img_with_boxes(config, img, scores, boxes, output_dir, epoch = None, name = None):
    if output_dir is None:
        output_dir = './output/'

    if epoch is None:
        epoch = ''

    if name is None:
        name = ''

    plt.figure(figsize=(16,10))
    plt.imshow(img)
    ax = plt.gca()
    if boxes is not None:    
        for p, (xmin, ymin, xmax, ymax), c in zip(scores, boxes.tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=1))
            cl = p.argmax()
            text = f": {p.item():.2f}"
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(output_dir + '/' + str(epoch) + str(name) + ".png", bbox_inches='tight', pad_inches=0)
    logging.info('Saved detection output to ' + output_dir + '/' + str(epoch) + str(name) + ".png")
    plt.close()

def plot_imgs(pil_imgs, output_dir = None, name = None):
    """
    Plots a variable number of input images side by side.

    Args:
        output_dir (str): Directory where the plot will be saved.
        epoch (int): Epoch number (for naming the plot).
        *pil_imgs (PIL.Image.Image): Variable number of input images (as PIL Image objects).

    Returns:
        None
    """
    if output_dir is None:
        output_dir = './output/'

    if name is None:
        name = ''

    plt.figure(figsize=(16, 10))
    num_imgs = len(pil_imgs)
    for i, pil_img in enumerate(pil_imgs, start=1):
        plt.subplot(1, num_imgs, i)
        #pil_img = np.transpose(pil_img[0], (1,2,0))
        plt.imshow(pil_img)
        plt.axis('off')

    # Save the plot
    #plot_filename = os.path.join(output_dir, f'epoch_{epoch}.png')

    plt.savefig(output_dir + '/' + str(name) + ".jpg", bbox_inches='tight', pad_inches=0)
    #plt.savefig(plot_filename, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_from_pkl(pkl_file: str, img_path: str, threshold: float = 0.01):
    detection_results = open_pkl_file(pkl_file)
    for i, image in enumerate(detection_results['images']):
        scores = detection_results['pred_scores'][i]
        pred_boxes = detection_results['pred_boxes'][i][scores>threshold]
        scores = scores[scores>threshold]
        image = utils.draw_bounding_boxes(image, pred_boxes)
        utils.save_image(image, img_path + '/' + str(i) + '.jpg')

def plot_img_with_boxes_and_gt(config, img_container, i = '', prefix = ''):

    img = img_container.converted_polar_image
    scores = img_container.scores
    boxes = img_container.boxes
    gt_boxes = img_container.polar_labels.boxes

    boxes = boxes[scores > config.POSTPROCESSING_SCORE]
    scores = scores[scores > config.POSTPROCESSING_SCORE]

    plt.figure(figsize=(16,10))
    plt.imshow(np.squeeze(img))
    ax = plt.gca()
    if boxes is not None:    
        for p, (xmin, ymin, xmax, ymax), c in zip(scores, boxes.tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=1))
            text = f"{p.item():.2f}"
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    if gt_boxes is not None:
        for (xmin, ymin, xmax, ymax) in gt_boxes.tolist():
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color='lime', linewidth=2))
            ax.text(xmin, ymin - 10, 'GT', fontsize=12, color='lime',
                    bbox=dict(facecolor='black', alpha=0.7))
    plt.axis('off')
    plt.savefig(config.OUTPUT_FOLDER + '/' + config.OUTPUT_IMAGEPREFIX + prefix + str(i) + ".png", bbox_inches='tight', pad_inches=0)
    logging.info('Saved detection output to ' + config.OUTPUT_FOLDER + '/' + config.OUTPUT_IMAGEPREFIX + prefix + str(i) + ".png")
    plt.close()