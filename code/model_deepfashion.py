import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

import permutation
from permutation import Conv1
#from skimage.transform import resize
import numpy as np
from torchvision.transforms import Resize
from torchvision.utils import save_image
import cv2


class FashionNet(nn.Module):
    def __init__(self, input_size):
        super(FashionNet, self).__init__()
        
        self.convmodel = nn.Sequential(

            # Conv1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Conv4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Conv5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Flatten()
        )

        self.tranmodel = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 300*200),
            nn.ReLU()
        )

        self.outmodel = nn.Sequential(
        
            # FC1
            nn.Linear(27648, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            # FC2
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),


            # Output
            nn.Linear(512, 46)
        )


    def forward(self, x1, x2):
        x2 = x2.view(x2.size(0), 300, 200)

        x1 = x1.view(x1.size(0), -1)
        x1 = self.tranmodel(x1)
        x1 = x1.view(x1.size(0), 300, 200)
        
        x = torch.zeros((x2.size(0), 1, 300, 200), dtype=torch.float32)
        for i in range(len(x2)):
            x[i, 0, :, :] = torch.mul(x2[i, :, :],x1[i, :, :])
        x = x.cuda()

        x = self.convmodel(x)
        y = self.outmodel(x)

        return y
