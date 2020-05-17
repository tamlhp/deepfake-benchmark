from pytorch_model.efficientnet import EfficientNet
import torch.nn as nn
from pytorch_model.efficientnet.model import Identity

model = EfficientNet.from_pretrained('efficientnet-b3',num_classes=1)

class EfficientPairwise(nn.Module):
    def __init__(self):
        super(EfficientPairwise, self).__init__()
        self.efficient = EfficientNet.from_pretrained('efficientnet-b3',num_classes=1)
        self.efficient._dropout = Identity()
        self.efficient._fc = Identity()
    def forward_once(self, x):
        output = self.efficient(x)

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

class EfficientFull(nn.Module):
    def __init__(self):
        super(EfficientFull, self).__init__()
        self.efficient = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1)
        self.efficient._dropout = Identity()
        self.efficient._fc = Identity()
        for param in self.cffn.parameters():
            param.requires_grad = False
        self.classify = nn.Linear(1536,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.efficient(input)
        x = self.classify(x)
        x = self.sigmoid(x)
        return x
