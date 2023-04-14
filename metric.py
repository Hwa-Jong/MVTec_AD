import numpy as np
import torch

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
