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

def train_capsule(train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',gpu_id=-1,manualSeed=0,resume="",beta1=0.9,dropout=0.05,image_size=256,batch_size=16,lr=0.003,num_workers=1,checkpoint="checkpoint",epochs=20):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    vgg_ext = VggExtractor().to(device)
    capnet = CapsuleNet(2,gpu_id=gpu_id).to(device)
    capsule_loss = CapsuleLoss().to(device)


    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # if gpu_id >= 0:
    #     torch.cuda.manual_seed_all(manualSeed)
    #     cudnn.benchmark = True

    if resume != "":
        text_writer = open(os.path.join(checkpoint, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(checkpoint, 'train.csv'), 'w')


    optimizer = Adam(capnet.parameters(), lr=lr, betas=(beta1, 0.999))

    if resume != "":
        capnet.load_state_dict(torch.load(os.path.join(checkpoint,'capsule_' + str(resume) + '.pt')))
        capnet.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint,'optim_' + str(resume) + '.pt')))

        if device != 'cpu':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    # if gpu_id >= 0:
    #     capnet.cuda(gpu_id)
    #     vgg_ext.cuda(gpu_id)
    #     capsule_loss.cuda(gpu_id)

    dataloader_train, dataloader_val = get_generate(train_set,val_set,image_size,batch_size,num_workers)
    capnet.train()
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

            img_data = img_data.to(device)
            labels_data = labels_data.to(device)
            # if gpu_id >= 0:
            #     img_data = img_data.cuda(gpu_id)
            #     labels_data = labels_data.cuda(gpu_id)

            input_v = Variable(img_data)
            x = vgg_ext(input_v)
            # x = x.cpu()
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
            count += batch_size


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
            count += batch_size

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count

        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.flush()
        capnet.train()

    text_writer.close()
    return


def eval_train(model ,dataloader_val,device,criterion,text_writer ):
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
    print(
          f"Test loss: {test_loss/len(dataloader_val):.3f}.. "
          f"Test accuracy: {accuracy/len(dataloader_val):.3f}")
    text_writer.write('Test loss %.4f, Test accuracy  %.4f \n' % (
        test_loss / len(dataloader_val), accuracy / len(dataloader_val)))
    text_writer.flush()
    model.train()
def train_cnn(model,train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',image_size=256,batch_size=16,resume = '',lr=0.003,num_workers=8,checkpoint="checkpoint",epochs=20,print_every=1000):
    from pytorch_model.focal_loss import FocalLoss
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True
    model = model.to(device)
    # criterion = nn.BCELoss().to(device)
    criterion = FocalLoss(gamma=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader_train, dataloader_val = get_generate(train_set,val_set,image_size,batch_size,num_workers)
    if resume != '':
        model.load_state_dict(torch.load( os.path.join(checkpoint, resume)))

    # train_losses, test_losses = [], []
    # import time
    text_writer = open(os.path.join(checkpoint, 'train.csv'), 'a')
    model.train()
    steps =0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in tqdm(dataloader_train):
            #     for inputs, labels in tqdm(testloader):
            steps += 1
            #         labels = np.array([labels])
            inputs, labels = inputs.to(device), labels.float().to(device)
            #         inputs, labels = inputs.to(device), labels[1].float().to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            logps = logps.squeeze()
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
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloader_val):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloader_val):.3f}")
                text_writer.write('Epoch %d, Train loss %.4f, Test loss %.4f, Test accuracy  %.4f \n' % (
                epoch, running_loss / print_every, test_loss / len(dataloader_val), accuracy / len(dataloader_val)))
                text_writer.flush()

                running_loss = 0
                steps = 0
                model.train()
        eval_train(model ,dataloader_val,device,criterion,text_writer)
        torch.save(model.state_dict(), os.path.join(checkpoint, 'model_pytorch_%d.pt' % epoch))
    return

def train_siamese(model,train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',image_size=256,length_embed = 1024,batch_size=16,resume = '',lr=0.001,num_workers=8,checkpoint="checkpoint",epochs=20,print_every=1000):
    from pytorch_model.siamese import ContrastiveLoss

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True

    model = model().to(device)
    criterion = ContrastiveLoss(length_embed)
    criterion2 = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    dataloader_train, dataloader_val = get_generate_siamese(train_set,val_set,image_size,batch_size,num_workers)


    if resume != '':
        model.load_state_dict(torch.load( os.path.join(checkpoint, resume)))
    text_writer = open(os.path.join(checkpoint, 'train.csv'), 'a')
    model.train()
    steps = 0
    running_loss_contrastive = 0
    running_loss_cls = 0
    for epoch in range(0, epochs):
        for img0, img1,y1,y2, label in tqdm(dataloader_train):
            steps += 1

            img0, img1,y1,y2, label = img0.to(device), img1.to(device),y1.float().to(device),y2.float().to(device), label.to(device)
            #         img0, img1 , label = img0, img1 , label
            # print(img0.size())
            optimizer.zero_grad()
            output1, output2,cls1,cls2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            running_loss_contrastive +=loss_contrastive.item()

            optimizer.zero_grad()
            output1, output2, cls1, cls2 = model(img0, img1)
            loss_cls1 = criterion2(cls1,y1)
            # loss_cls1.backward()
            # optimizer.step()
            # print(loss_cls1)
            # optimizer.zero_grad()
            loss_cls2 = criterion2(cls2,y2)
            # loss_cls2.backward()
            loss_cls = loss_cls1 + loss_cls2
            loss_cls.backward()
            optimizer.step()
            running_loss_cls += loss_cls
            if steps % print_every == 0:
                test_loss_contrastive = 0
                test_loss_cls = 0
                accuracy1 = 0
                accuracy2 = 0
                model.eval()
                # print("Epoch number {}\n iteration_number {} Current loss {}\n".format(epoch, iteration_number,
                #                                                                        loss_contrastive.item()))
                with torch.no_grad():
                    for img0,img1,y1,y2, label in dataloader_val:
                        img0,img1, y1, y2, label = img0.to(device), img1.to(device),\
                                                    y1.float().to(device), y2.float().to(device), label.to(device)
                        output1, output2, cls1, cls2 = model.forward(img0, img1)
                        cls1 = cls1.squeeze()
                        cls2 = cls2.squeeze()
                        batch_loss_contrastive = criterion(output1, output2, label)
                        #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
                        test_loss_contrastive += batch_loss_contrastive.item()
                        loss_cls1 = criterion2(cls1, y1)
                        loss_cls2 = criterion2(cls2, y2)
                        test_loss_cls += loss_cls1 + loss_cls2

                        #                     print("labels : ",labels)
                        #                     print("logps  : ",logps)
                        equals1 = y1 == (cls1 > 0.5)
                        equals2 = y2 == (cls2 > 0.5)
                        #                     print("equals   ",equals)
                        accuracy1 += torch.mean(equals1.type(torch.FloatTensor)).item()
                        accuracy2 += torch.mean(equals2.type(torch.FloatTensor)).item()
                #                 train_losses.append(running_loss/len(trainloader))
                #             test_losses.append(test_loss/len(testloader))
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss contrastive: {running_loss_contrastive/print_every:.3f}.. "
                      f"Train loss cls: {running_loss_cls/print_every:.3f}.. "
                      f"Test loss contrastive: {test_loss_contrastive/len(dataloader_val):.3f}.. "
                      f"Test loss cls: {test_loss_cls/len(dataloader_val):.3f}.. "
                      f"Test accuracy 1: {accuracy1/len(dataloader_val):.3f}"
                      f"Test accuracy 2: {accuracy2/len(dataloader_val):.3f}")
                text_writer.write('Epoch %d, Train loss contrastive %.4f,Train loss cls %.4f  , Test loss contrastive %.4f,Test loss cls %.4f,\
                 Test accuracy 1  %.4f ,Test accuracy 2  %.4f \n' % (
                epoch, running_loss_contrastive / print_every,running_loss_cls / print_every, \
                test_loss_contrastive / len(dataloader_val),test_loss_cls / len(dataloader_val),\
                accuracy1 / len(dataloader_val),accuracy2 / len(dataloader_val) ))
                text_writer.flush()

                steps = 0
                running_loss_contrastive = 0
                running_loss_cls = 0
                model.train()
        torch.save(model.state_dict(), os.path.join(checkpoint, 'model_pytorch_%d.pt' % epoch))

