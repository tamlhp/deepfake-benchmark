from pytorch_model.wavelet_model.conv_block import BasicConv,\
    DAB,WaveletPool,\
    TransformerBlock
import torch.nn as nn
from pytorch_model.wavelet_model import dct
class WaveletModel(nn.Module):
    def __init__(self,in_channel):
        super(WaveletModel, self).__init__()
        self.in_channel = in_channel
        # self.dct = dct.dct
        self.pool = WaveletPool()
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.out1 = 16
        self.conv1 = BasicConv(in_planes=4*in_channel,out_planes=self.out1)
        self.dab1 = DAB(self.out1,kernel_size=3,reduction=4)
        self.out2 = 64
        self.conv2 = BasicConv(in_planes=4*self.out1,out_planes=self.out2)
        self.dab2 = DAB(self.out2,kernel_size=3,reduction=4)
        self.out3 = 128
        self.conv3 = BasicConv(in_planes=4*self.out2,out_planes=self.out3)
        self.dab3 = DAB(self.out3,kernel_size=3,reduction=8)
        self.out4 = 256
        self.conv4 = BasicConv(in_planes=4*self.out3,out_planes=self.out4)
        self.dab4 = DAB(self.out4,kernel_size=3,reduction=8)

        self.conv5 = BasicConv(in_planes=self.out4,out_planes=self.out4)
        self.dab5 = DAB(self.out4,kernel_size=3,reduction=8)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.out4,self.out2)
        self.fc2 = nn.Linear(self.out2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        bs = inputs.size(0)
        inputs = dct.dct_2d(inputs)
        # print(inputs.size())
        x = self.pool(inputs)
        x = self.conv1(x)
        # x = self.dab1(x)
        x = self.pool(x)
        x = self.conv2(x)
        # x = self.dab2(x)
        x = self.pool(x)
        x = self.conv3(x)
        # x = self.dab3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.dab4(x)
        x = self.conv5(x)
        # x = self.dab5(x)
        out = self._avg_pooling(x)
        out = out.view(bs, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
# import torch
if __name__ == "__main__":

    model = WaveletModel(in_channel=3)
    # from torchsummary import summary_string
    import torchsummary
    import sys
    # print(sys.stdout)
    # sys.stdout = open("test.txt", "w")

    sum = torchsummary.summary(model,(3,256,256))
    # result, params_info = summary_string(model,(3,256,256),batch_size=-1, device=torch.device('cuda:0'), dtypes=None)
    # print(sum)
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    # print("aaa")

