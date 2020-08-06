import glob
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
class PairwiseDataset(Dataset):

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
            ID = self.df_path[index]
        else:
            ID = self.real_path[index]
        X_l, X_r, y = self.__data_generation(ID, rr)

        return X_l, X_r,y

    def __data_generation(self, ID, rr):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # Store sample
        img = Image.open(ID)
        while len(img.mode) < 3:
            rr = random.randint(0, 1)
            if rr == 0:
                ID = random.choice(self.real_path)
            else:
                ID = random.choice(self.df_path)
            img = Image.open(ID)
        rr2 = random.randint(0, 1)
        if rr2 == 0:
            ID2 = random.choice(self.real_path)
        else:
            ID2 = random.choice(self.df_path)
        img2 = Image.open(ID2)

        while len(img2.mode) < 3:
            rr = random.randint(0, 1)
            if rr == 0:
                ID2 = random.choice(self.real_path)
            else:
                ID2 = random.choice(self.df_path)
            img2 = Image.open(ID2)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        X_l = img
        X_r = img2
        # Store class
        y = 1 if rr == rr2 else 0
        # X = [X_l, X_r]
        # y = [y1,y2,y]
        return X_l, X_r,y
    def __len__(self):
        return int(np.floor(min(len(self.df_path), len(self.real_path))))


def get_generate_pairwise(train_set,image_size,batch_size,num_workers):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomApply([
                                               transforms.RandomRotation(5),
                                               transforms.RandomAffine(degrees=5, scale=(0.95, 1.05))
                                           ], p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])

                                           ])
    pairwise_dataset = PairwiseDataset(path=train_set,
                                            transform=transform_fwd
                                            , should_invert=False,shuffle=True)
    print("pairwise_dataset len :   ",pairwise_dataset.__len__())


    assert pairwise_dataset

    pairwise_dataset = torch.utils.data.DataLoader(pairwise_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    return pairwise_dataset