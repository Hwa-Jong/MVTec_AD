import numpy as np
import torch

from mvtec_ad_evaluation.roc_curve_util import compute_classification_roc
from mvtec_ad_evaluation.generic_util import trapezoid

def calculate_IoU(label, pred):
    '''
    label : label image (uint8 0~255)
    pred : prediction image (uint8 0~255)

    0 : background
    1 : defect
    '''
    intersection = torch.logical_and(label, pred)
    union = torch.logical_or(label, pred)
    iou = torch.sum(intersection, dim=[1,2,3]) / (torch.sum(union, dim=[1,2,3]) + 1e-6)

    return iou


def calc_classification_roc(anomaly_maps, scoring_function, ground_truth_labels):
    all_fprs, all_tprs = compute_classification_roc(anomaly_maps, scoring_function, ground_truth_labels)
    return trapezoid(all_fprs, all_tprs)