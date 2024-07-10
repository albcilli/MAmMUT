import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

import permutation
from permutation import Conv1
import numpy as np
from torchvision.transforms import Resize
from torchvision.utils import save_image
import cv2
from torchvision import models


#Transformation Network Classifier, tabular data is "encoded" in a syntethic image using a FF-Neural Network.
#The obtained image is again combined with the original one with an Hadamard product to inlclude in it new information. A CNN classifier is the applied.
class TranClassNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.convmodel = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size = 3, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(128, 64, kernel_size = 3, stride = 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Flatten()
        )
        self.outmodel = nn.Sequential(
            nn.Linear(43776, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(200, 18)
        )
        self.tranmodel = nn.Sequential(
            nn.Linear(input_size, 3600),
            nn.BatchNorm1d(3600),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(3600, 178*384),
            nn.ReLU()
        )


    def forward(self, x1, x2):
        x1 = x1.view(x1.size(0), -1)
        x1 = self.tranmodel(x1)

        x1 = x1.view(x1.size(0), 178, 384)
        x2 = x2.view(x2.size(0), 178, 384)

        x = torch.zeros((x2.size(0), 1, 178, 384), dtype=torch.float32)
        for i in range(len(x2)):
            x[i, 0, :, :] = torch.mul(x2[i, :, :],x1[i, :, :])
        x = x.cuda()
        x = self.convmodel(x)
        y = self.outmodel(x)

        return y


class ResClassNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        new_first_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1)
        num_features = pretrained_model.fc.in_features
        pretrained_model = nn.Sequential(new_first_layer, *list(pretrained_model.children())[:-1])

        for param in pretrained_model.parameters():
            param.requires_grad = True

        self.convmodel = nn.Sequential(
            pretrained_model,
            nn.Flatten()
        )

        self.outmodel = nn.Sequential(
            nn.Linear(num_features, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(200, 18)
        )
        self.tranmodel = nn.Sequential(
            nn.Linear(input_size, 3600),
            nn.BatchNorm1d(3600),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(3600, 178*384),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = x1.view(x1.size(0), -1)
        x1 = self.tranmodel(x1)

        x1 = x1.view(x1.size(0), 178, 384)
        x2 = x2.view(x2.size(0), 178, 384)

        x = torch.zeros((x2.size(0), 1, 178, 384), dtype=torch.float32)
        for i in range(len(x2)):
            x[i, 0, :, :] = torch.mul(x2[i, :, :],x1[i, :, :])
            
        x = x.cuda()
        x = self.convmodel(x)
        y = self.outmodel(x)

        return y
