import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import models, transforms, utils
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
from collections import OrderedDict
import os
import sys
import glob
import time

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.featex import FeatureExtractor
from utils import get_children, model_features
from engine.nms import nms
from engine.plot_box import plot_box


parser = argparse.ArgumentParser(description='template matching using CNN')
parser.add_argument('image_path')
parser.add_argument('template_path')
args = parser.parse_args()
    
# DEFINE DATA AND LABEL
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

model_feature = nn.Sequential(*get_children(model_features()))
    
FE = FeatureExtractor(model_feature, use_cuda=False, padding=True, print_layer=True)

image_path = args.image_path
template_path = args.template_path

start = time.time()
image_name = os.path.basename(image_path)
raw_image = cv2.imread(image_path)[..., ::-1]
image = image_transform(raw_image.copy()).unsqueeze(0)
d_img = raw_image.astype(np.uint8).copy()   # d_img is used for drawing

template_name = os.path.basename(template_path)
raw_template = cv2.imread(template_path)[..., ::-1]
template = image_transform(raw_template.copy()).unsqueeze(0)

boxes, _, scores = FE(template, image, threshold=None)
if len(boxes) < 1 or len(scores) < 1:
    exit()

nms_res = nms(np.array(boxes), np.array(scores), thresh=0.7)
print("nms_res:", nms_res)
d_img = plot_box(d_img, boxes, nms_res)

if cv2.imwrite("result.jpg", d_img[..., ::-1]):
    print("result.jpg was generated")
    
end = time.time()
    
print("Time:", round(end - start, 2))

