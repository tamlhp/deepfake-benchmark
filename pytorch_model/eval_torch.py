import torch
import random
import os
import torchvision.transforms as transforms
from torch.optim import Adam
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from sklearn import metrics
import numpy as np
from torch.autograd import Variable
from pytorch_model.capsule_pytorch.model import VggExtractor,CapsuleNet,CapsuleLoss
from torch import optim
import torch.nn as nn

from tqdm import tqdm

def get_generate(val_set,image_size,batch_size,num_workers):
    transform_fwd = transforms.Compose([transforms.Resize(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                           ])

    dataset_val = datasets.ImageFolder(val_set,
                                     transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)

    return dataloader_val


def eval_capsule(val_set ='../../extract_raw_img',resume=0,image_size=256,batch_size=16,num_workers=1,checkpoint="checkpoint"):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    vgg_ext = VggExtractor().to(device)
    capnet = CapsuleNet(2).to(device)
    capsule_loss = CapsuleLoss().to(device)

    capnet.load_state_dict(torch.load(os.path.join(checkpoint,'capsule_' + str(resume) + '.pt')))


    dataloader_val = get_generate(val_set,image_size,batch_size,num_workers)

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    count = 0
    loss_test = 0
    for img_data, labels_data in dataloader_val:

        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float)

        img_data = img_data.to(device)
        labels_data = labels_data.to(device)
        # if gpu_id >= 0:
        #     img_data = img_data.cuda(gpu_id)
        #     labels_data = labels_data.cuda(gpu_id)

        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=False)

        loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
        loss_dis_data = loss_dis.item()
        output_dis = class_.data.cpu().numpy()

        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        loss_test += loss_dis_data
        count += 1

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count
    print('Test loss: %.4f  acc: %.2f'% ( loss_test, acc_test*100))

    return

def eval_cnn(model,val_set ='../../extract_raw_img',image_size=256,resume=0,batch_size=16,num_workers=8,checkpoint="checkpoint"):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss().to(device)

    model.load_state_dict(torch.load( os.path.join(checkpoint, 'mnasnet_pytorch_%d.pt' % resume)))

    dataloader_val = get_generate(val_set,image_size,batch_size,num_workers)

    # train_losses, test_losses = [], []
    # import time
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader_val:
            inputs, labels = inputs.to(device), labels.float().to(device)
            logps = model.forward(inputs)
            logps = logps.squeeze()
            batch_loss = criterion(logps, labels)
            #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
            test_loss += batch_loss.item()
            #                     print("labels : ",labels)
            #                     print("logps  : ",logps)
            equals = labels == (logps > 0.5)
            #                     print("equals   ",equals)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #                 train_losses.append(running_loss/len(trainloader))
    #             test_losses.append(test_loss/len(testloader))
    print(f"Test loss: {test_loss/len(dataloader_val):.3f}.. "
          f"Test accuracy: {accuracy/len(dataloader_val):.3f}")


    return

