import torch
import torchvision.transforms as transforms
# import torchtoolbox.transform as transforms

import torchvision.datasets as datasets
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image,ImageEnhance
import cv2
from albumentations.augmentations.transforms import ImageCompression, GaussNoise,GaussianBlur,Resize,HorizontalFlip,Rotate,ShiftScaleRotate
from albumentations import Compose,Normalize
from albumentations import pytorch as AT

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_generate(train_set,val_set,image_size,batch_size,num_workers):
    # transforms_2 = Compose([Resize(image_size, image_size),
    #                         ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    #                         HorizontalFlip(p=0.5),
    #                         Rotate(limit=5,p=0.5),
    #                         ShiftScaleRotate(shift_limit=0,scale_limit=0.05,rotate_limit=5,p=0.5),
    #                         GaussNoise(p=0.1),
    #                         GaussianBlur(blur_limit=3, p=0.05),
    #                         Normalize(mean=[0.485, 0.456, 0.406],
    #                                             std=[0.229, 0.224, 0.225]),
    #                         AT.ToTensorV2()])
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
    transform_fwd_test = transforms.Compose([transforms.Resize((image_size, image_size)),
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
                                     transform=transform_fwd_test)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)

    return dataloader_train,dataloader_val
def get_jpeg_augmentation():
    train_transform = [
        ImageCompression(quality_lower=50, quality_upper=51, p=1.0)
    ]
    transforms =  Compose(train_transform)
    return lambda img:transforms(image=np.array(img))['image']

def get_val_generate(val_set,image_size,batch_size,num_workers,adj_brightness=1.0, adj_contrast=1.0):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),
                                        # transforms.Resize((int(image_size/2), int(image_size/2))),
                                        # transforms.Resize((image_size, image_size)),
                                        # transforms.RandomGaussianNoise(p=0.0),
                                        AddGaussianNoise(0, 10),
                                        transforms.RandomErasing(),
                                        transforms.Lambda(lambda img :transforms.functional.adjust_brightness(img,adj_brightness)),
                                        transforms.Lambda(lambda img :transforms.functional.adjust_contrast(img,adj_contrast)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                           #transforms.RandomErasing(p=1.0,scale=(0.5,0.5001),ratio=(1,1.0001))
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

#=============================================================
#**************************************************************
class ImageGeneratorDualFFT(Dataset):

    def __init__(self, path,image_size, transform=None,transform_fft = None, should_invert=True,shuffle=True,adj_brightness=None, adj_contrast=None):
        self.path = path
        self.transform = transform
        self.image_size =image_size
        self.transform_fft = transform_fft
        self.should_invert = should_invert
        self.shuffle = shuffle
        data_path = []
        data_path = data_path + glob.glob(path + "/*/*.jpg")
        data_path = data_path + glob.glob(path + "/*/*.jpeg")
        data_path = data_path + glob.glob(path + "/*/*.png")
        self.data_path = data_path
        np.random.shuffle(self.data_path)
        self.indexes = range(len(self.data_path))
        self.on_epoch_end()
        self.adj_brightness = adj_brightness
        self.adj_contrast = adj_contrast

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
    def __getitem__(self, index):

        img = cv2.imread(self.data_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.image_size,self.image_size))
        if self.adj_brightness is not None and self.adj_contrast is not None:
            PIL_img1 = Image.fromarray(img)
            enhancer = ImageEnhance.Brightness(PIL_img1)
            img_adj = enhancer.enhance(self.adj_brightness)
            enhancer = ImageEnhance.Contrast(img_adj)
            img_adj = enhancer.enhance(self.adj_contrast)
            img = np.array(img_adj)

        # PIL_img = Image.fromarray(img)
        # if self.transform is not None:
        #     PIL_img = self.transform(PIL_img)
        #     img = np.array(transforms_ori.ToPILImage()(PIL_img))

        f = np.fft.fft2(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        fshift = np.fft.fftshift(f)
        fshift += 1e-8

        magnitude_spectrum = np.log(np.abs(fshift))
        # img = np.concatenate([img,magnitude_spectrum],axis=2)
        # img = np.transpose(img,(2,0,1))
        magnitude_spectrum = cv2.resize(magnitude_spectrum,(self.image_size,self.image_size))
        magnitude_spectrum = np.array([magnitude_spectrum])
        magnitude_spectrum = np.transpose(magnitude_spectrum, (1,2 , 0))
        PIL_img = Image.fromarray(img)
        # PIL_magnitude_spectrum = Image.fromarray(magnitude_spectrum)
        if self.transform is not None:
            PIL_img = self.transform(PIL_img)
        if self.transform_fft is not None:
            magnitude_spectrum = self.transform_fft(magnitude_spectrum)

        y = 0
        if '0_real' in self.data_path[index]:
            y = 0
        elif '1_df' in self.data_path[index] or '1_f2f' in self.data_path[index] or '1_fs' in self.data_path[index] or '1_nt' in self.data_path[index]:
            y = 1
        return PIL_img,magnitude_spectrum,y

    def __len__(self):
        return int(np.floor(len(self.data_path)))
def get_generate_dualfft(train_set,val_set,image_size,batch_size,num_workers):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),

                                        # transforms.Resize((int(image_size/2), int(image_size/2))),
                                        # transforms.Resize((image_size, image_size)),
                                        # transforms.RandomGaussianNoise(p=0.0),

                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),


                                           ])
    transform_fft = transforms.Compose([transforms.ToTensor()])
    fft_dataset = ImageGeneratorDualFFT(path=train_set,image_size= image_size,
                                            transform=transform_fwd,transform_fft = transform_fft
                                            , should_invert=False,shuffle=True)
    print("fft dual len :   ",fft_dataset.__len__())


    assert fft_dataset
    dataset_train = datasets.ImageFolder(train_set,
                                      transform=transform_fwd)
    assert dataset_train
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader_train = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, sampler=sampler,
                                              num_workers=num_workers)
    dataset_val = ImageGeneratorDualFFT(path=val_set,image_size=image_size,
                                            transform=transform_fwd,transform_fft = transform_fft
                                            , should_invert=False,shuffle=True)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)
    return dataloader_train,dataloader_val

def get_val_generate_dualfft(train_set,image_size,batch_size,num_workers,adj_brightness=1.0, adj_contrast=1.0):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])

                                           ])
    transform_fft = transforms.Compose([transforms.ToTensor()])
    fft_dataset = ImageGeneratorDualFFT(path=train_set,image_size=image_size,
                                            transform=transform_fwd,transform_fft=transform_fft
                                            , should_invert=False,shuffle=True,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
    print("fft dual len :   ",fft_dataset.__len__())


    assert fft_dataset

    fft_dataloader = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    return fft_dataloader
#**************************************************************
#=============================================================

#=============================================================
#**************************************************************
class ImageGeneratorFFT(Dataset):

    def __init__(self, path,image_size,transform_fft = None, should_invert=True,shuffle=True,adj_brightness=None, adj_contrast=None):
        self.path = path
        self.image_size =image_size
        self.transform_fft = transform_fft
        self.should_invert = should_invert
        self.shuffle = shuffle
        data_path = []
        data_path = data_path + glob.glob(path + "/*/*.jpg")
        data_path = data_path + glob.glob(path + "/*/*.jpeg")
        data_path = data_path + glob.glob(path + "/*/*.png")
        self.data_path = data_path

        self.indexes = range(len(self.data_path))
        np.random.shuffle(self.data_path)
        self.on_epoch_end()
        self.adj_brightness = adj_brightness
        self.adj_contrast = adj_contrast
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
    def __getitem__(self, index):
        img = cv2.imread(self.data_path[index],0)
        img = cv2.resize(img, (self.image_size, self.image_size))
        if self.adj_brightness is not None and self.adj_contrast is not None:
            PIL_img1 = Image.fromarray(img)
            enhancer = ImageEnhance.Brightness(PIL_img1)
            img_adj = enhancer.enhance(self.adj_brightness)
            enhancer = ImageEnhance.Contrast(img_adj)
            img_adj = enhancer.enhance(self.adj_contrast)
            img = np.array(img_adj)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fshift += 1e-8

        magnitude_spectrum = np.log(np.abs(fshift))
        # img = np.concatenate([img,magnitude_spectrum],axis=2)
        # img = np.transpose(img,(2,0,1))
        magnitude_spectrum = cv2.resize(magnitude_spectrum,(self.image_size,self.image_size))
        magnitude_spectrum = np.array([magnitude_spectrum])
        min_magnitude_spectrum = np.min(magnitude_spectrum)
        max_magnitude_spectrum = np.max(magnitude_spectrum)
        magnitude_spectrum = (magnitude_spectrum -min_magnitude_spectrum ) / (max_magnitude_spectrum-min_magnitude_spectrum)
        magnitude_spectrum = np.transpose(magnitude_spectrum, (1,2 , 0))
        # PIL_magnitude_spectrum = Image.fromarray(magnitude_spectrum)
        if self.transform_fft is not None:
            magnitude_spectrum = self.transform_fft(magnitude_spectrum)

        y = 0
        if '0_real' in self.data_path[index]:
            y = 0
        elif '1_df' in self.data_path[index] or '1_f2f' in self.data_path[index] or '1_fs' in self.data_path[
                index] or '1_nt' in self.data_path[index]:

            y = 1
        return magnitude_spectrum,y

    def __len__(self):
        return int(np.floor(len(self.data_path)))
def get_generate_fft(train_set,val_set,image_size,batch_size,num_workers):

    transform_fft = transforms.Compose([transforms.ToTensor()])
    fft_dataset = ImageGeneratorFFT(path=train_set,image_size= image_size,
                                            transform_fft = transform_fft
                                            , should_invert=False,shuffle=True)
    print("fft dual len :   ",fft_dataset.__len__())


    assert fft_dataset
    dataset_train = datasets.ImageFolder(train_set)
    assert dataset_train
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader_train = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, sampler=sampler,
                                              num_workers=num_workers)
    dataset_val = ImageGeneratorFFT(path=val_set,image_size=image_size,
                                           transform_fft = transform_fft
                                            , should_invert=False,shuffle=True)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)
    return dataloader_train,dataloader_val

def get_val_generate_fft(train_set,image_size,batch_size,num_workers,adj_brightness=1.0, adj_contrast=1.0):

    transform_fft = transforms.Compose([transforms.ToTensor()])
    fft_dataset = ImageGeneratorFFT(path=train_set,image_size=image_size,
                                            transform_fft=transform_fft
                                            , should_invert=False,shuffle=True,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
    print("fft dual len :   ",fft_dataset.__len__())


    assert fft_dataset

    fft_dataloader = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    return fft_dataloader
#**************************************************************
#=============================================================


#=============================================================
#**************************************************************
class ImageGenerator4dFFT(Dataset):

    def __init__(self, path,image_size,transform = None, should_invert=True,shuffle=True,adj_brightness=None, adj_contrast=None):
        self.path = path
        self.image_size =image_size
        self.transform = transform
        self.should_invert = should_invert
        self.shuffle = shuffle
        data_path = []
        data_path = data_path + glob.glob(path + "/*/*.jpg")
        data_path = data_path + glob.glob(path + "/*/*.jpeg")
        data_path = data_path + glob.glob(path + "/*/*.png")
        self.data_path = data_path

        self.indexes = range(len(self.data_path))
        np.random.shuffle(self.data_path)
        self.on_epoch_end()
        self.adj_brightness = adj_brightness
        self.adj_contrast = adj_contrast
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
    def __getitem__(self, index):

        img = cv2.imread(self.data_path[index])
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.adj_brightness is not None and self.adj_contrast is not None:
            PIL_img1 = Image.fromarray(img)
            enhancer = ImageEnhance.Brightness(PIL_img1)
            img_adj = enhancer.enhance(self.adj_brightness)
            enhancer = ImageEnhance.Contrast(img_adj)
            img_adj = enhancer.enhance(self.adj_contrast)
            img = np.array(img_adj)
        f = np.fft.fft2(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        fshift = np.fft.fftshift(f)
        fshift += 1e-8

        magnitude_spectrum = np.log(np.abs(fshift))
        # img = np.concatenate([img,magnitude_spectrum],axis=2)
        # img = np.transpose(img,(2,0,1))
        magnitude_spectrum = cv2.resize(magnitude_spectrum,(self.image_size,self.image_size))
        magnitude_spectrum = np.array([magnitude_spectrum])
        min_magnitude_spectrum = np.min(magnitude_spectrum)
        max_magnitude_spectrum = np.max(magnitude_spectrum)
        magnitude_spectrum = (magnitude_spectrum -min_magnitude_spectrum ) / (max_magnitude_spectrum-min_magnitude_spectrum)
        magnitude_spectrum = np.transpose(magnitude_spectrum, (1,2 , 0))
        img = img/255.0
        # PIL_magnitude_spectrum = Image.fromarray(magnitude_spectrum)
        img = np.concatenate([img, magnitude_spectrum], axis=2)
        if self.transform is not None:
            img = self.transform(img)

        y = 0
        if '0_real' in self.data_path[index]:
            y = 0
        elif '1_df' in self.data_path[index] or '1_f2f' in self.data_path[index] or '1_fs' in self.data_path[
                index] or '1_nt' in self.data_path[index]:

            y = 1
        return img,y

    def __len__(self):
        return int(np.floor(len(self.data_path)))
def get_generate_4dfft(train_set,val_set,image_size,batch_size,num_workers):

    transform = transforms.Compose([transforms.ToTensor()])
    fft_dataset = ImageGenerator4dFFT(path=train_set,image_size= image_size,
                                      transform = transform
                                            , should_invert=False,shuffle=True)
    print("fft dual len :   ",fft_dataset.__len__())


    assert fft_dataset
    dataset_train = datasets.ImageFolder(train_set)
    assert dataset_train
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader_train = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, sampler=sampler,
                                              num_workers=num_workers)
    dataset_val = ImageGenerator4dFFT(path=val_set,image_size=image_size,
                                      transform = transform
                                            , should_invert=False,shuffle=True)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)
    return dataloader_train,dataloader_val

def get_val_generate_4dfft(train_set,image_size,batch_size,num_workers,adj_brightness=1.0, adj_contrast=1.0):
    transform = transforms.Compose([transforms.ToTensor()])
    fft_dataset = ImageGenerator4dFFT(path=train_set,image_size=image_size,
                                      transform=transform
                                            , should_invert=False,shuffle=True,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
    print("fft dual len :   ",fft_dataset.__len__())


    assert fft_dataset

    fft_dataloader = torch.utils.data.DataLoader(fft_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    return fft_dataloader
#**************************************************************
#=============================================================
