import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, sys
import progressbar
import torch.nn as nn
import torchvision.transforms as transforms

from utils import compute_score, all_sample_iou, plot_success_curve, IoU
from model.featex import FeatureExtractor
from utils import get_children, model_features
from engine.nms import nms
from engine.plot_box import plot_box


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])
    
# load data
file_dir = 'datasets/BBSdata'

gt = sorted([ os.path.join(file_dir, i) for i in os.listdir(file_dir)  if '.txt' in i ])
img_path = sorted([ os.path.join(file_dir, i) for i in os.listdir(file_dir) if '.jpg' in i ] )
def read_gt(file_path):
    with open(file_path) as IN:
        x, y, w, h = [ eval(i) for i in IN.readline().strip().split(',')]
    return x, y, w, h

def model_eval(FE):
    num_samples = len(img_path) // 2
    gt_list, score_list = [], []
    num_false_image = 0
    for idx in range(num_samples):
        # get template
        template_raw = cv2.imread(img_path[2*idx])[...,::-1]
        template_bbox = read_gt(gt[2*idx])
        x, y, w, h = [int(round(t)) for t in template_bbox]
        template = template_raw[y:y+h, x:x+w]
        
        image = cv2.imread(img_path[2*idx+1])[...,::-1]
        image_name = os.path.basename(img_path[2*idx+1])
        image_gt = read_gt(gt[2*idx+1])
        x_gt, y_gt, w_gt, h_gt = [int(round(t)) for t in image_gt]
        d_img = image.astype(np.uint8).copy()

        # process images
        template_ = image_transform(template.copy()).unsqueeze(0)
        image_ = image_transform(image.copy()).unsqueeze(0)
        
        # sub_fol = os.path.join("result", image_name.split('.')[0])
        # os.mkdir(sub_fol)
        # cv2.imwrite(os.path.join(sub_fol, "template.jpg"), template[..., ::-1])
        # cv2.imwrite(os.path.join(sub_fol, "image.jpg"), image[..., ::-1])
        
        boxes, _, scores = FE(template_, image_, threshold=None)
        if len(boxes) == 0:
            num_false_image += 1
            d_img = cv2.rectangle(d_img, (x_gt, y_gt), (x_gt+w_gt, y_gt+h_gt), (0, 255, 0), 1)
            cv2.imwrite("result/{}".format(image_name), d_img[..., ::-1])
            continue
        
        x_pd, y_pd, w_pd, h_pd = boxes[0][0][0], boxes[0][0][1], boxes[0][1][0]-boxes[0][0][0], boxes[0][1][1]-boxes[0][0][1]
        pred_bbox = [x_pd, y_pd, w_pd, h_pd]
        d_img = plot_box(d_img, boxes)
        d_img = cv2.rectangle(d_img, (x_gt, y_gt), (x_gt+w_gt, y_gt+h_gt), (0, 255, 0), 1)
        
        score_list.append(pred_bbox)
        gt_list.append(image_gt)
        
        if cv2.imwrite("result/{}".format(image_name), d_img[..., ::-1]):
            print("result/{} was generated".format(image_name))
        
    return score_list, gt_list
    
# Create model
model_feature = nn.Sequential(*get_children(model_features()))
FE = FeatureExtractor(model_feature, use_cuda=False, padding=True)

score_list, gt_list = model_eval(FE)
iou_score = all_sample_iou( score_list, gt_list )
plot_success_curve( iou_score, title='he2 ')