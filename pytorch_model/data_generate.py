import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


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


