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


def train_capsule(manualSeed,gpu_id,resume,checkpoint,lr,beta1,imageSize,train_set,val_set,batch_size,num_workers,epochs,dropout ):
    vgg_ext = VggExtractor()
    capnet = CapsuleNet(2, gpu_id)
    capsule_loss = CapsuleLoss(gpu_id)


    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    if gpu_id >= 0:
        torch.cuda.manual_seed_all(manualSeed)
        cudnn.benchmark = True

    if resume > 0:
        text_writer = open(os.path.join(checkpoint, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(checkpoint, 'train.csv'), 'w')




    optimizer = Adam(capnet.parameters(), lr=lr, betas=(beta1, 0.999))

    if resume > 0:
        capnet.load_state_dict(torch.load(os.path.join(checkpoint,'capsule_' + str(resume) + '.pt')))
        capnet.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint,'optim_' + str(resume) + '.pt')))

        if gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(gpu_id)

    if gpu_id >= 0:
        capnet.cuda(gpu_id)
        vgg_ext.cuda(gpu_id)
        capsule_loss.cuda(gpu_id)

    transform_fwd = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
           transforms.RandomRotation(5),
           transforms.RandomAffine(degrees=5,scale=(0.95,1.05))
           ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset_train = datasets.ImageFolder(root=train_set, transform=transform_fwd)
    assert dataset_train
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=int(num_workers))

    dataset_val = datasets.ImageFolder(root=val_set, transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=int(num_workers))


    for epoch in range(resume+1, epochs+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(dataloader_train):

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float)
            optimizer.zero_grad()

            if gpu_id >= 0:
                img_data = img_data.cuda(gpu_id)
                labels_data = labels_data.cuda(gpu_id)

            input_v = Variable(img_data)
            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=random, dropout=dropout)

            loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            optimizer.step()

            output_dis = class_.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_train += loss_dis_data
            count += 1


        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count

        ########################################################################

        # do checkpointing & validation
        torch.save(capnet.state_dict(), os.path.join(checkpoint, 'capsule_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint, 'optim_%d.pt' % epoch))

        capnet.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        count = 0

        for img_data, labels_data in dataloader_val:

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float)

            if gpu_id >= 0:
                img_data = img_data.cuda(gpu_id)
                labels_data = labels_data.cuda(gpu_id)

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

        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.flush()
        capnet.train(mode=True)

    text_writer.close()
    return

def train_cnn(model,batch_size,num_workers,checkpoint,epochs,print_every=1000  ):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    criterion = nn.BCELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.003)


    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomApply([
                                               transforms.RandomRotation(5),
                                               transforms.RandomAffine(degrees=5, scale=(0.95, 1.05))
                                           ], p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])

                                           ])
    dataset_train = datasets.ImageFolder('/data/tam/kaggle/extract_raw_img',
                                      transform=train_transforms)
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler,
                                              num_workers=num_workers)

    dataset_val = datasets.ImageFolder('/data/tam/kaggle/extract_raw_img_test',
                                     transform=train_transforms)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)



    # train_losses, test_losses = [], []
    # import time
    text_writer = open(os.path.join(checkpoint, 'train.csv'), 'a')
    model.train()
    steps =0
    for epoch in range(epochs):
        for inputs, labels in tqdm(dataloader_train):
            #     for inputs, labels in tqdm(testloader):
            steps += 1
            #         labels = np.array([labels])
            inputs, labels = inputs.to(device), labels.float().to(device)
            #         inputs, labels = inputs.to(device), labels[1].float().to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)[:, 0]
            loss = criterion(logps, labels)
            #         loss = F.binary_cross_entropy_with_logits(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # time.sleep(0.05)
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader_val:
                        inputs, labels = inputs.to(device), labels.float().to(device)
                        logps = model.forward(inputs)[:, 0]
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
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloader_val):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloader_val):.3f}")
                text_writer.write('Epoch %d, Train loss %.4f, Test loss %.4f, Test accuracy  %.4f \n' % (
                epoch, running_loss / print_every, test_loss / len(dataloader_val), accuracy / len(dataloader_val)))
                text_writer.flush()

                running_loss = 0
                model.train()
        torch.save(model.state_dict(), os.path.join(checkpoint, 'mnasnet_pytorch_%d.pt' % epoch))
    return

