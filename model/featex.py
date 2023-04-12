import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
import numpy as np
import copy
import time
import torch.nn.functional as Fu
from sklearn.metrics.pairwise import manhattan_distances, cosine_distances
from scipy.spatial.distance import minkowski, chebyshev, canberra, correlation
from sklearn.decomposition import PCA

from utils import reduce_dimensional_tensor
        
class FeatureExtractor():
    def __init__(self, model, use_cuda=True, padding=True, print_layer=False):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        self.use_cuda = use_cuda
        self.feature_maps = []
        self.is_print_layer = print_layer

        if self.use_cuda:
            self.model = self.model.cuda()

        self.index = []
        self.f = []
        self.stride = []
        
        for i, module in enumerate(self.model.children()):
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                self.index.append(i)
                self.f.append(module.kernel_size[0])
                self.stride.append(module.stride[0])
            if isinstance(module, nn.MaxPool2d):
                if padding:
                    module.padding = 1
                self.index.append(i)
                self.f.append(module.kernel_size)
                self.stride.append(module.stride)

        self.rf = np.array(self.calc_rf(self.f, self.stride))

    def save_template_feature_map1(self, module, input, output):
        self.template_feature_map1 = output.detach()
        
    def save_template_feature_map2(self, module, input, output):
        self.template_feature_map2 = output.detach()

    def save_image_feature_map1(self, module, input, output):
        self.image_feature_map1 = output.detach()
        
    def save_image_feature_map2(self, module, input, output):
        self.image_feature_map2 = output.detach()


    def calc_rf(self, f, stride):
        rf = []
        for i in range(len(f)):
            if i == 0:
                rf.append(3)
            else:
                rf.append(rf[i-1] + (f[i]-1)*self.product(stride[:i]))
        return rf

    def product(self, lis):
        if len(lis) == 0:
            return 0
        else:
            res = 1
            for x in lis:
                res *= x
            return res

    def calc_l_star(self, template, k=2):
        l = np.sum(self.rf <= min(list(template.size()[-2:]))) - 1
        l_star = max(l - k, 2)
        return l_star
    
    def calc_NCC(self, F, M):
        c, h_f, w_f = F.shape[-3:]
        NCC = np.zeros((M.shape[-2] - h_f, M.shape[-1] - w_f))
        F_tilde = F.reshape(1, -1)
        
        for i in range(M.shape[-2] - h_f):
            for j in range(M.shape[-1] - w_f):
                # self.pca.fit(M[:, :, i:i + h_f, j:j + w_f].reshape(c, -1))
                # M_tilde = self.pca.singular_values_.reshape(1, -1)
                M_tilde = M[:, :, i:i + h_f, j:j + w_f].reshape(1, -1)
                
                NCC[i, j] = np.sum(cosine_distances(F_tilde, M_tilde))

        return NCC

    def __call__(self, template, image, threshold=None):
        if self.use_cuda:
            template = template.cuda()
            image = image.cuda()

        self.l_star = self.calc_l_star(template)
        layer_1 = self.l_star - 2

        # save template feature map (named F in paper)
        template_handle1 = self.model[self.index[layer_1]].register_forward_hook(
            self.save_template_feature_map1)
        self.model(template)
        template_handle1.remove()

        template_handle2 = self.model[self.index[self.l_star]].register_forward_hook(
            self.save_template_feature_map2)
        self.model(template)
        template_handle2.remove()
        
        if self.is_print_layer:
            print("=============================================================")
            print("self.template_feature_map1: ", self.template_feature_map1.shape)
            print("self.template_feature_map2: ", self.template_feature_map2.shape)
            
        
        # interpolate feature map 2 to same size of feature map 1
        self.template_feature_map2 = Fu.interpolate(self.template_feature_map2,
                                                   size=(self.template_feature_map1.size()[2], self.template_feature_map1.size()[3]),
                                                   mode='bicubic', align_corners=True)
            
        self.template_feature_map1 = reduce_dimensional_tensor(self.template_feature_map1,
                                                               step=int(self.template_feature_map1.shape[1] / 2))
        self.template_feature_map2 = reduce_dimensional_tensor(self.template_feature_map2, 
                                                               step=int(self.template_feature_map2.shape[1] / 8))
        
        if self.is_print_layer:
            print("self.template_feature_map1 after: ", self.template_feature_map1.shape)
            print("self.template_feature_map2 after: ", self.template_feature_map2.shape)
        
        self.template_feature_map = torch.cat((self.template_feature_map1, self.template_feature_map2), dim=1)


        # save image feature map (named M in paper)
        image_handle1 = self.model[self.index[layer_1]].register_forward_hook(
            self.save_image_feature_map1)
        self.model(image)
        image_handle1.remove()
        
        image_handle2 = self.model[self.index[self.l_star]].register_forward_hook(
            self.save_image_feature_map2)
        self.model(image)
        image_handle2.remove()
        
        # interpolate feature map 2 to same size of feature map 1
        self.image_feature_map2 = Fu.interpolate(self.image_feature_map2,
                                                size=(self.image_feature_map1.size()[2], self.image_feature_map1.size()[3]),
                                                mode='bicubic', align_corners=True)
        
        self.image_feature_map1 = reduce_dimensional_tensor(self.image_feature_map1, 
                                                            step=int(self.image_feature_map1.shape[1] / 2))
        self.image_feature_map2 = reduce_dimensional_tensor(self.image_feature_map2, 
                                                            step=int(self.image_feature_map2.shape[1] / 8))
        
        self.image_feature_map = torch.cat((self.image_feature_map1, self.image_feature_map2), dim=1)
        
        if self.is_print_layer:
            print("=============================================================")
            print("Chosse layer 1:", self.index[layer_1])
            print("Choose layer 2:", self.index[self.l_star])
            print("template_feature_map:", self.template_feature_map.size())
            print("image_feature_map:", self.image_feature_map.size())
            print("=============================================================")
            self.is_print_layer = False
        

        if self.use_cuda:
            self.template_feature_map = self.template_feature_map.cpu()
            self.image_feature_map = self.image_feature_map.cpu()

        # calc NCC
        F = self.template_feature_map.numpy()[0].astype(np.float32)
        M = self.image_feature_map.numpy()[0].astype(np.float32)

        self.NCC = self.calc_NCC(
            self.template_feature_map.numpy(), self.image_feature_map.numpy())

        if threshold is None:
            threshold = 1.01 * np.min(self.NCC)
        max_indices = np.array(np.where(self.NCC < threshold)).T

        boxes = []
        centers = []
        scores = []
        for max_index in max_indices:
            i_star, j_star = max_index
            if i_star < 2 or j_star < 2 or i_star >= self.NCC.shape[0] - 2 or j_star >= self.NCC.shape[1] - 2:
                continue
            
            NCC_part = self.NCC[i_star-1:i_star+2, j_star-2:j_star+2]

            x_center = (j_star + self.template_feature_map.size()
                        [-1]/2) * image.size()[-1] // self.image_feature_map.size()[-1]
            y_center = (i_star + self.template_feature_map.size()
                        [-2]/2) * image.size()[-2] // self.image_feature_map.size()[-2]

            x1_0 = x_center - template.size()[-1]/2
            x2_0 = x_center + template.size()[-1]/2
            y1_0 = y_center - template.size()[-2]/2
            y2_0 = y_center + template.size()[-2]/2

            stride_product = self.product(self.stride[:self.l_star])
            
            x1 = np.sum(
                NCC_part * (x1_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / np.sum(NCC_part)
            x2 = np.sum(
                NCC_part * (x2_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / np.sum(NCC_part)
            y1 = np.sum(
                NCC_part * (y1_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / np.sum(NCC_part)
            y2 = np.sum(
                NCC_part * (y2_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / np.sum(NCC_part)

            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))
            x_center = int(round(x_center))
            y_center = int(round(y_center))

            boxes.append([(x1, y1), (x2, y2)])
            centers.append((x_center, y_center))
            scores.append(np.sum(NCC_part))
        
        return boxes, centers, scores