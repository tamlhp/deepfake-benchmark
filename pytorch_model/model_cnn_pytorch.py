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
        self.fc = nn.Sequential(nn.Linear(2048, 1),
                                 nn.Sigmoid())

def MyResNetX():
    model = MyResNeXt()
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
class Meso4(nn.Module):
	"""
	Pytorch Implemention of Meso4
	Autor: Honggu Liu
	Date: July 4, 2019
	"""
	def __init__(self, num_classes=2):
		super(Meso4, self).__init__()
		self.num_classes = num_classes
		self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(8)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)

		self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
		self.bn2 = nn.BatchNorm2d(16)
		self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
		self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
		#flatten: x = x.view(x.size(0), -1)
		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*8*8, 16)
		self.fc2 = nn.Linear(16, num_classes)

	def forward(self, input):
		x = self.conv1(input) #(8, 256, 256)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(8, 128, 128)

		x = self.conv2(x) #(8, 128, 128)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(8, 64, 64)

		x = self.conv3(x) #(16, 64, 64)
		x = self.relu(x)
		x = self.bn2(x)
		x = self.maxpooling1(x) #(16, 32, 32)

		x = self.conv4(x) #(16, 32, 32)
		x = self.relu(x)
		x = self.bn2(x)
		x = self.maxpooling2(x) #(16, 8, 8)

		x = x.view(x.size(0), -1) #(Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x) #(Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x
def mesonet():
	model = Meso4()
	return model
if __name__ == '__main__':
    # model = resnext50(False)
    model = mesonet()
    import torchsummary
    torchsummary.summary(model, (3, 256, 256))