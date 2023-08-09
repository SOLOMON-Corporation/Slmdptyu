import os
import cv2
from slmutils.annotator import Annotator


def plot_batch(class_names, images, file_names, batch_results, save_dir, plot_cfg):
    annotator = Annotator(
        show_bboxes=plot_cfg['show_bboxes'], 
        show_masks=plot_cfg['show_masks'], 
        show_kpts=plot_cfg['show_kpts'], 
        show_scores=plot_cfg['show_scores'], 
        show_label=plot_cfg['show_label'],
        class_names=class_names, 
        colors=None,
    )
    for image, results, file_name in zip(images, batch_results, file_names):
        cv2.imwrite(
            os.path.join(save_dir, os.path.basename(file_name)),
            annotator(image, results)
        )

def plot(image, results, class_names, plot_cfg):
    annotator = Annotator(
        show_bboxes=plot_cfg['show_bboxes'], 
        show_masks=plot_cfg['show_masks'], 
        show_kpts=plot_cfg['show_kpts'], 
        show_scores=plot_cfg['show_scores'], 
        show_label=plot_cfg['show_label'],
        class_names=class_names, 
        colors=None,
    )
    return annotator(image, results)