import os, sys, random

import torch
import torch.nn as nn

import torchvision.models as models

class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)

        # self.load_state_dict(checkpoint)

        # Override the existing FC layer with a new one.
        self.fc = nn.Linear(2048, 1)

def MyResNetX():
    model = models.resnet.ResNet()
    model.fc = nn.Sequential(nn.Linear(2048, 1),
                                 nn.Sigmoid())
    return model
def resnext50(pretrained=True):
    model = models.resnext50_32x4d(pretrained=pretrained)
    model.fc = nn.Sequential(nn.Linear(2048, 1),
                                 nn.Sigmoid())
    return model

def mnasnet(pretrained=True):
    model = models.mnasnet1_0(pretrained=pretrained)
    model.classifier = nn.Sequential(nn.Linear(1280, 1),
                                     nn.Sigmoid())
    return model

def resnext101(pretrained=True):
    model = models.resnext101_32x8d(pretrained=pretrained)
    model.fc = nn.Sequential(nn.Linear(2048, 1),
                             nn.Sigmoid())
    return model