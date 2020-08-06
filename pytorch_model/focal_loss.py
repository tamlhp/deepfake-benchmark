import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# https://github.com/clcarwin/focal_loss_pytorch
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
if __name__ == "__main__":
    import os, sys, random, time

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    start_time = time.time()
    maxe = 0
    for i in range(2):
        x = torch.rand(12800, 1) * random.randint(1, 10)
        print(x)
        x = Variable(x.to(device))
        l = torch.rand(12800).ge(0.1).long()
        l = Variable(l.to(device))

        output0 = FocalLoss(gamma=0)(x, l)
        output1 = nn.CrossEntropyLoss()(x, l)
        # print(output0)
        a = output0.item()
        b = output1.item()
        if abs(a - b) > maxe: maxe = abs(a - b)
    print('time:', time.time() - start_time, 'max_error:', maxe)

    start_time = time.time()
    maxe = 0
    for i in range(1):
        x = torch.rand(128, 1000, 8, 4) * random.randint(1, 10)
        x = Variable(x.to(device))
        l = torch.rand(128, 8, 4) * 1000  # 1000 is classes_num
        l = l.long()
        l = Variable(l.to(device))

        output0 = FocalLoss(gamma=0)(x, l)
        output1 = nn.NLLLoss2d()(F.log_softmax(x), l)
        a = output0.item()
        b = output1.item()
        if abs(a - b) > maxe: maxe = abs(a - b)
    print('time:', time.time() - start_time, 'max_error:', maxe)
"""
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit =input
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_pros = -1 * target * torch.log(logit) # cross entropy
        loss_cons = -1 * (1-target) * torch.log(1-logit) # cross entropy
        loss = loss_pros * (1 - logit) ** self.gamma + loss_cons*(logit)**self.gamma # focal loss

        return loss.sum()


if __name__ == "__main__":
    import os, sys, random, time

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    start_time = time.time()
    maxe = 0
    for i in range(2):
        x = torch.rand(100) * random.randint(0, 1)
        print(x)
        x = Variable(x.to(device))
        l = torch.rand(100).ge(0.1).float()
        l = Variable(l.to(device))
        print(l)
        output0 = FocalLoss(gamma=2)(1-l, l)
        output1 = nn.BCELoss()(1-l, l)
        # print(output0)
        a = output0.item()
        b = output1.item()
        print(a)
        print(b)
        if abs(a - b) > maxe: maxe = abs(a - b)
    print('time:', time.time() - start_time, 'max_error:', maxe)
