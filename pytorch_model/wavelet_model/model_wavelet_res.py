from pytorch_model.wavelet_model.conv_block import BasicConv,\
    DAB,WaveletPool,\
    TransformerBlock
import torch.nn as nn
import torch
from pytorch_model.wavelet_model import dct
class WaveletResModel(nn.Module):
    def __init__(self,in_channel):
        super(WaveletResModel, self).__init__()
        self.in_channel = in_channel
        # self.dct = dct.dct
        self.pool = WaveletPool()
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.out1 = 16
        self.conv1 = BasicConv(in_planes=4*in_channel,out_planes=self.out1)
        # self.dab1 = DAB(self.out1,kernel_size=3,reduction=4)
        self.out2 = 8
        self.conv2 = BasicConv(in_planes=4*self.out1,out_planes=self.out2)
        self.out2_1 = 16
        self.conv2_1 = BasicConv(in_planes=4*4*4*in_channel,out_planes=self.out2_1)

        self.dab3 = DAB(4*self.out2 + self.out2_1 ,kernel_size=3,reduction=16)
        self.out3 = 16
        self.conv3 = BasicConv(in_planes=4*self.out2 + self.out2_1,out_planes=self.out3)
        self.out3_1 = 32
        self.conv3_1 = BasicConv(in_planes=4*4*4*self.out1, out_planes=self.out3_1)

        self.dab4 = DAB(4*self.out3+ self.out3_1 ,kernel_size=3,reduction=32)
        self.out4 = 32
        self.conv4 = BasicConv(in_planes=4*self.out3+ self.out3_1,out_planes=self.out4)
        self.out4_1 = 64
        self.conv4_1 = BasicConv(in_planes=4*4*4*self.out2, out_planes=self.out4_1)


        self.dab5 = DAB(4*self.out4+ self.out4_1,kernel_size=3,reduction=32)

        # self.conv5 = BasicConv(in_planes=self.out4,out_planes=self.out4)
        # self.dab5 = DAB(self.out4,kernel_size=3,reduction=8)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(4*self.out4+ self.out4_1,self.out2)
        self.fc2 = nn.Linear(self.out2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        bs = inputs.size(0)
        # inputs = dct.dct_2d(inputs)
        # print(inputs.size())
        x0 = self.pool(inputs)
        x1 = self.conv1(x0)
        # x = self.dab1(x)
        x1 = self.pool(x1)


        x2 = self.conv2(x1)
        # x = self.dab2(x)
        x2 = self.pool(x2)
        # print(x2.size())

        x3 = torch.cat([x2,self.conv2_1(self.pool(self.pool(x0)))],1)
        x3 = self.dab3(x3)
        x3 = self.conv3(x3)
        # x = self.dab3(x)
        x3 = self.pool(x3)


        x4 = torch.cat([x3, self.conv3_1(self.pool(self.pool(x1)))], 1)
        x4 = self.dab4(x4)
        x4 = self.conv4(x4)
        # x = self.dab4(x)
        x4 = self.pool(x4)


        x5 = torch.cat([x4, self.conv4_1(self.pool(self.pool(x2)))], 1)
        x5 = self.dab5(x5)

        out = self._avg_pooling(x5)
        out = out.view(bs, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
# import torch
if __name__ == "__main__":

    model = WaveletResModel(in_channel=3)
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

