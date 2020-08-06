import os
from torch.utils.data import DataLoader, Dataset

import numpy as np
import random
from PIL import Image
import torch
# from torch.autograd import Variable
# import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import glob
from torchvision import transforms, datasets, models

class SiameseNetworkDataset(Dataset):

    def __init__(self, path, transform=None, should_invert=True,shuffle=True):
        self.path = path
        self.transform = transform
        self.should_invert = should_invert
        self.shuffle = shuffle
        df_path = []
        df_path = df_path + glob.glob(path + "/*df/*.jpg")
        df_path = df_path + glob.glob(path + "/*df/*.jpeg")
        df_path = df_path + glob.glob(path + "/*df/*.png")
        self.df_path = df_path
        real_path = []
        real_path = real_path + glob.glob(path + "/*real/*.jpg")
        real_path = real_path + glob.glob(path + "/*real/*.jpeg")
        real_path = real_path +glob.glob(path + "/*real/*.png")
        self.real_path = real_path
        self.indexes = range(min(len(self.df_path), len(self.real_path)))
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.df_path)
            np.random.shuffle(self.real_path)
    def __getitem__(self, index):
        rr = random.randint(0, 1)
        if rr == 0:
            ID = self.real_path[index]
        else:
            ID = self.df_path[index]
        X_l, X_r, y1, y2, y = self.__data_generation(ID, rr)

        return X_l, X_r,y1,y2,y

    def __data_generation(self, ID, rr):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # Store sample
        img = Image.open(ID)
        rr2 = random.randint(0, 1)
        if rr2 == 0:
            ID2 = random.choice(self.real_path)
        else:
            ID2 = random.choice(self.df_path)
        img2 = Image.open(ID2)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        X_l = img
        X_r = img2
        y1 = int(rr)
        y2 = int(rr2)
        # Store class
        y = 1 if rr == rr2 else 0
        # X = [X_l, X_r]
        # y = [y1,y2,y]
        return X_l, X_r,y1,y2,y
    def __len__(self):
        return int(np.floor(min(len(self.df_path), len(self.real_path))))

class SiameseNetworkResnet(nn.Module):
    def __init__(self,length_embed,pretrained=False):
        super(SiameseNetworkResnet, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, length_embed)
        self.model.cls = torch.nn.Linear(length_embed, 1)
        self.model.soft = nn.Sigmoid()
    def forward_once(self, x):
        output = self.model(x)

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        cls1 = self.model.cls(output1)
        cls1 = self.model.soft(cls1)
        cls2 = self.model.cls(output2)
        cls2 = self.model.soft(cls2)
        return output1, output2,cls1,cls2




class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self,device, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.max(torch.tensor(0.0).to(self.device),torch.pow(torch.tensor(self.margin).to(self.device) - euclidean_distance, 2)))
        return loss_contrastive

if __name__ == "__main__":
    IMGWIDTH = 128

    training_dir = "../../../extract_raw_img"
    testing_dir = "../../../extract_raw_img"
    train_batch_size = 2
    train_number_epochs = 1


    checkpoint = "checkpoint"
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    siamese_dataset = SiameseNetworkDataset(path=training_dir,
                                            transform=transforms.Compose([transforms.Resize((IMGWIDTH, IMGWIDTH)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False,shuffle=True)
    print("114 :   ",siamese_dataset.__len__())


    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=1,
                                  batch_size=train_batch_size)
    device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    model = SiameseNetworkResnet(length_embed=128).to(device)
    criterion = ContrastiveLoss(device)
    criterion2 = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    import time
    from tqdm import tqdm
    iteration_number = 0
    for epoch in range(0, train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1,y1,y2, label = data
            img0, img1,y1,y2, label = img0.to(device), img1.to(device),y1.float().to(device),y2.float().to(device), label.to(device)
            print(i)
            # print(y1,"   ",y2,"   ",label)
            # continue
            #         img0, img1 , label = img0, img1 , label
            # print(img0.size())
            optimizer.zero_grad()
            output1, output2,cls1,cls2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            optimizer.zero_grad()
            output1, output2, cls1, cls2 = model(img0, img1)
            loss_cls1 = criterion2(cls1,y1)
            # loss_cls1.backward()
            # optimizer.step()
            print(loss_cls1)

            optimizer.zero_grad()
            loss_cls2 = criterion2(cls2,y2)
            # loss_cls2.backward()
            loss_cls = loss_cls1 + loss_cls2
            loss_cls.backward()
            optimizer.step()
            if i % 2 == 0:
                print("Epoch number {}\n iteration_number {} Current loss {}\n".format(epoch, iteration_number,
                                                                                       loss_contrastive.item()))
                iteration_number += 2
                # counter.append(iteration_number)
                # loss_history.append(loss_contrastive.item())
                if i % 2 == 0:
                    torch.save(model.state_dict(), os.path.join(checkpoint, 'pairwise_%d.pt' % epoch))
    torch.save(model.state_dict(), os.path.join(checkpoint, 'pairwise_end.pt'))

