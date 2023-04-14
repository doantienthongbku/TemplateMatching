from functools import reduce
import torch
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import convolve
from sklearn.metrics import auc

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.model import PPLCNet_x2_5

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def model_features():
    model = PPLCNet_x2_5()
    model.load_state_dict(torch.load("weights/PPLCNet_x2_5_pretrained.pth"))

    stem = model.conv1
    features = [stem, model.blocks2, model.blocks3, model.blocks4, model.blocks5, model.blocks6]

    return torch.nn.Sequential(*features)


def reduce_dimensional_tensor(input: torch.Tensor, step: int = 4):
    _, c, h, w = input.size()
    feature_list = []
    for start_c in range(0, c, step):
        temp_tensor = torch.max(input[:, start_c:start_c + step, :, :], dim=1)
        temp_tensor = temp_tensor.values.unsqueeze(1)
        feature_list.append(temp_tensor)
        
    return torch.cat(feature_list, dim=1)

def IoU( r1, r2 ):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1; y12 = y11 + h1
    x22 = x21 + w2; y22 = y21 + h2
    x_overlap = max(0, min(x12,x22) - max(x11,x21) )
    y_overlap = max(0, min(y12,y22) - max(y11,y21) )
    I = 1. * x_overlap * y_overlap
    U = (y12-y11)*(x12-x11) + (y22-y21)*(x22-x21) - I
    J = I/U
    return J

def evaluate_iou( rect_gt, rect_pred ):
    # score of iou
    score = [ IoU(i, j) for i, j in zip(rect_gt, rect_pred) ]
    return score

def compute_score( x, w, h ):
    # score of response strength
    k = np.ones( (h, w) )
    score = convolve( x, k, mode='wrap' )
    score[:, :w//2] = 0
    score[:, math.ceil(-w/2):] = 0
    score[:h//2, :] = 0
    score[math.ceil(-h/2):, :] = 0
    return score

def locate_bbox( a, w, h ):
    row = np.argmax( np.max(a, axis=1) )
    col = np.argmax( np.max(a, axis=0) )
    x = col - 1. * w / 2
    y = row - 1. * h / 2
    return x, y, w, h

def score2curve( score, thres_delta = 0.01 ):
    thres = np.linspace( 0, 1, int(1./thres_delta)+1 )
    success_num = []
    for th in thres:
        success_num.append( np.sum(score >= (th+1e-6)) )
    success_rate = np.array(success_num) / len(score)
    return thres, success_rate

    return bboxes

def all_sample_iou(score_list, gt_list):
    num_samples = len(score_list)
    iou_list = []
    for idx in range(num_samples):
        score, image_gt = score_list[idx], gt_list[idx]
        iou = IoU( image_gt, score )
        iou_list.append( iou )
    return iou_list


def plot_success_curve( iou_score, title='' ):
    thres, success_rate = score2curve( iou_score, thres_delta = 0.05 )
    auc_ = np.mean( success_rate[:-1] ) # this is same auc protocol as used in previous template matching papers #auc_ = auc( thres, success_rate ) # this is the actual auc
    plt.figure()
    plt.grid(True)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0, 1)
    plt.title(title + 'auc={}'.format(auc_))
    plt.plot( thres, success_rate )
    plt.savefig(title + '.png')
    

if __name__ == '__main__':
    example = torch.rand(1, 320, 19, 19)
    out = reduce_dimensional_tensor(example, step=4)
    print(out.size())
    
