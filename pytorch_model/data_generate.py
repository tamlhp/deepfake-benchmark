import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2

# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    print(count)
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    print(weight_per_class)
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_generate(train_set,val_set,image_size,batch_size,num_workers):
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
    dataset_train = datasets.ImageFolder(train_set,
                                      transform=transform_fwd)
    assert dataset_train
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler,
                                              num_workers=num_workers)

    dataset_val = datasets.ImageFolder(val_set,
                                     transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)

    return dataloader_train,dataloader_val

def get_val_generate(val_set,image_size,batch_size,num_workers):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                           ])

    dataset_val = datasets.ImageFolder(val_set,
                                     transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)

    return dataloader_val

def get_generate_siamese(train_set,val_set,image_size,batch_size,num_workers):
    from pytorch_model.siamese import SiameseNetworkDataset

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
    dataset_train = SiameseNetworkDataset(path=train_set,
                                            transform=transform_fwd
                                            , should_invert=False,shuffle=True)

    assert dataset_train

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)

    dataset_val = SiameseNetworkDataset(path=val_set,
                                            transform=transform_fwd
                                            , should_invert=False,shuffle=True)

    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)

    return dataloader_train,dataloader_val


class ImageGeneratorFFT(Dataset):

    def __init__(self, path, transform=None,transformfft = None, should_invert=True,shuffle=True):
        self.path = path
        self.transform = transform
        self.transformfft = transformfft
        self.should_invert = should_invert
        self.shuffle = shuffle
        data_path = []
        data_path = data_path + glob.glob(path + "/*/*.jpg")
        data_path = data_path + glob.glob(path + "/*/*.jpeg")
        data_path = data_path + glob.glob(path + "/*/*.png")
        self.data_path = data_path

        self.indexes = range(len(self.data_path))
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
    def __getitem__(self, index):

        img = cv2.imread(self.data_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (128, 128))
        f = np.fft.fft2(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        fshift = np.fft.fftshift(f)
        fshift += 1e-8

        magnitude_spectrum = np.log(np.abs(fshift))
        magnitude_spectrum = np.array([magnitude_spectrum]).T
        # img = np.concatenate([img,magnitude_spectrum],axis=2)
        img = np.transpose(img,(2,0,1))
        magnitude_spectrum = np.transpose(magnitude_spectrum, (2, 0, 1))
        if self.transform is not None:
            img = self.transform(img)
        if self.transformfft is not None:
            magnitude_spectrum = self.transformfft(magnitude_spectrum)

        y = 0
        if 'real' in self.data_path[index]:
            y = 0
        elif 'df' in self.data_path[index]:
            y = 1
        return img,magnitude_spectrum,y

    def __len__(self):
        return int(np.floor(len(self.data_path)))
def get_generate_fft(train_set,val_set,image_size,batch_size,num_workers):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])

                                           ])
    transform_fft = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.ToTensor()])
    fft_dataset = ImageGeneratorFFT(path=train_set,
                                            transform=transform_fwd,transform_fft = transform_fft
                                            , should_invert=False,shuffle=True)
    print("pairwise_dataset len :   ",fft_dataset.__len__())


    assert fft_dataset
    dataset_train = datasets.ImageFolder(train_set,
                                      transform=transform_fwd)
    assert dataset_train
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader_train = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, sampler=sampler,
                                              num_workers=num_workers)
    dataset_val = ImageGeneratorFFT(path=val_set,
                                            transform=transform_fwd
                                            , should_invert=False,shuffle=True)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)
    return dataloader_train,dataloader_val

def get_val_generate_fft(train_set,image_size,batch_size,num_workers):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])

                                           ])
    transform_fft = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.ToTensor()])
    fft_dataset = ImageGeneratorFFT(path=train_set,
                                            transform=transform_fwd,transform_fft=transform_fft
                                            , should_invert=False,shuffle=True)
    print("pairwise_dataset len :   ",fft_dataset.__len__())


    assert fft_dataset

    fft_dataloader = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    return fft_dataloader