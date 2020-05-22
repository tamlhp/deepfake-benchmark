import torch
import random
import os

from sklearn import metrics
import numpy as np
from torch.autograd import Variable
from pytorch_model.capsule_pytorch.model import VggExtractor,CapsuleNet,CapsuleLoss
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import recall_score,accuracy_score,precision_score,log_loss,classification_report
from pytorch_model.data_generate import get_val_generate,get_val_generate_dualfft,get_val_generate_fft,get_val_generate_4dfft




def eval_capsule(val_set ='../../extract_raw_img',gpu_id=-1,resume=0,image_size=256,batch_size=16,num_workers=1,checkpoint="checkpoint",show_time=False):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    vgg_ext = VggExtractor().to(device)
    capnet = CapsuleNet(2,gpu_id)
    capnet = capnet.to(device)
    capsule_loss = CapsuleLoss().to(device)
    # optimizer = Adam(capnet.parameters(), lr=0.003, betas=(0.9, 0.999))

    capnet.load_state_dict(torch.load(os.path.join(checkpoint,'capsule_' + str(resume) + '.pt')))


    dataloader_val = get_val_generate(val_set,image_size,batch_size,num_workers)

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    count = 0
    loss_test = 0
    y_label = []
    y_pred = []
    y_pred_label = []
    capnet.eval()
    for img_data, labels_data in tqdm(dataloader_val):
        begin = time.time()
        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float)

        img_data = img_data.to(device)
        labels_data = labels_data.to(device)
        # if gpu_id >= 0:
        #     img_data = img_data.cuda(gpu_id)
        #     labels_data = labels_data.cuda(gpu_id)

        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        # x = x.cpu()
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
        if show_time:
            print("Time : ", time.time()-begin)
        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        loss_test += loss_dis_data
        count += batch_size
        y_label.extend(img_label)
        y_pred.extend(output_dis)
        y_pred_label.extend(output_pred)
    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count
    print('Test loss: %.4f  acc: %.2f'% ( loss_test, acc_test*100))
    log_loss_metric = log_loss(y_label, y_pred, labels=np.array([0., 1.]))
    print(f"Test log_loss: {log_loss_metric:.3f}\n" +
          f"Test accuracy_score: {accuracy_score(y_label,y_pred_label):.3f}\n" +
          f"Test precision_score: {precision_score(y_label,y_pred_label):.3f}\n" +
          f"Test recall: {recall_score(y_label,y_pred_label):.3f}\n")
    print(classification_report(y_label,y_pred_label))
    return

def eval_cnn(model,val_set ='../../extract_raw_img',image_size=256,resume="",batch_size=16,num_workers=8,checkpoint="checkpoint",show_time=False):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss().to(device)

    model.load_state_dict(torch.load(os.path.join(checkpoint, resume)))

    dataloader_val = get_val_generate(val_set,image_size,batch_size,num_workers)

    # train_losses, test_losses = [], []
    # import time
    test_loss = 0
    accuracy = 0
    model.eval()
    y_label = []
    y_pred = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_val):
            begin = time.time()
            # print(labels)
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs, labels = inputs.to(device), labels.float().to(device)
            logps = model.forward(inputs)
            logps = logps.squeeze()
            # print(logps)
            logps_cpu = logps.cpu().numpy()
            y_pred.extend(logps_cpu.astype(np.float64))
            if show_time:
                print("Time : ", time.time() - begin)
            batch_loss = criterion(logps, labels)
            #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
            test_loss += batch_loss.item()
            #                     print("labels : ",labels)
            #                     print("logps  : ",logps)
            equals = labels == (logps > 0.5)
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            #                     print("equals   ",equals)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #                 train_losses.append(running_loss/len(trainloader))
    #             test_losses.append(test_loss/len(testloader))
    print(f"Test loss: {test_loss/len(dataloader_val):.3f}.. \n" +
          f"Test accuracy: {accuracy/len(dataloader_val):.3f}\n" +
          f"Test log_loss: {log_loss(y_label,y_pred,labels=np.array([0.,1.])):.3f}\n" +
          f"Test accuracy_score: {accuracy_score(y_label,y_pred_label):.3f}\n" +
          f"Test precision_score: {precision_score(y_label,y_pred_label):.3f}\n" +
          f"Test recall: {recall_score(y_label,y_pred_label):.3f}\n")
    print(classification_report(y_label,y_pred_label))

    return

def eval_dualcnn(model,val_set ='../../extract_raw_img',image_size=256,resume="",batch_size=16,num_workers=8,checkpoint="checkpoint",show_time=False):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss().to(device)

    model.load_state_dict(torch.load(os.path.join(checkpoint, resume)))

    dataloader_val = get_val_generate_dualfft(val_set,image_size,batch_size,num_workers)

    # train_losses, test_losses = [], []
    # import time
    test_loss = 0
    accuracy = 0
    model.eval()
    y_label = []
    y_pred = []
    y_pred_label = []
    with torch.no_grad():
        for inputs,img_fft, labels in tqdm(dataloader_val):
            begin = time.time()
            # print(labels)
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs,img_fft, labels = inputs.float().to(device),img_fft.float().to(device), labels.float().to(device)
            logps = model.forward(inputs,img_fft)
            logps = logps.squeeze()
            # print(logps)
            logps_cpu = logps.cpu().numpy()
            y_pred.extend(logps_cpu.astype(np.float64))
            if show_time:
                print("Time : ", time.time() - begin)
            batch_loss = criterion(logps, labels)
            #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
            test_loss += batch_loss.item()
            #                     print("labels : ",labels)
            #                     print("logps  : ",logps)
            equals = labels == (logps > 0.5)
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            #                     print("equals   ",equals)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #                 train_losses.append(running_loss/len(trainloader))
    #             test_losses.append(test_loss/len(testloader))
    print(f"Test loss: {test_loss/len(dataloader_val):.3f}.. \n" +
          f"Test accuracy: {accuracy/len(dataloader_val):.3f}\n" +
          f"Test log_loss: {log_loss(y_label,y_pred,labels=np.array([0.,1.])):.3f}\n" +
          f"Test accuracy_score: {accuracy_score(y_label,y_pred_label):.3f}\n" +
          f"Test precision_score: {precision_score(y_label,y_pred_label):.3f}\n" +
          f"Test recall: {recall_score(y_label,y_pred_label):.3f}\n")
    print(classification_report(y_label,y_pred_label))

    return

def eval_fftcnn(model,val_set ='../../extract_raw_img',image_size=256,resume="",batch_size=16,num_workers=8,checkpoint="checkpoint",show_time=False):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss().to(device)

    model.load_state_dict(torch.load(os.path.join(checkpoint, resume)))

    dataloader_val = get_val_generate_fft(val_set,image_size,batch_size,num_workers)

    # train_losses, test_losses = [], []
    # import time
    test_loss = 0
    accuracy = 0
    model.eval()
    y_label = []
    y_pred = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_val):
            begin = time.time()
            # print(labels)
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            logps = model.forward(inputs)
            logps = logps.squeeze()
            # print(logps)
            logps_cpu = logps.cpu().numpy()
            y_pred.extend(logps_cpu.astype(np.float64))
            if show_time:
                print("Time : ", time.time() - begin)
            batch_loss = criterion(logps, labels)
            #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
            test_loss += batch_loss.item()
            #                     print("labels : ",labels)
            #                     print("logps  : ",logps)
            equals = labels == (logps > 0.5)
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            #                     print("equals   ",equals)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #                 train_losses.append(running_loss/len(trainloader))
    #             test_losses.append(test_loss/len(testloader))
    print(f"Test loss: {test_loss/len(dataloader_val):.3f}.. \n" +
          f"Test accuracy: {accuracy/len(dataloader_val):.3f}\n" +
          f"Test log_loss: {log_loss(y_label,y_pred,labels=np.array([0.,1.])):.3f}\n" +
          f"Test accuracy_score: {accuracy_score(y_label,y_pred_label):.3f}\n" +
          f"Test precision_score: {precision_score(y_label,y_pred_label):.3f}\n" +
          f"Test recall: {recall_score(y_label,y_pred_label):.3f}\n")
    print(classification_report(y_label,y_pred_label))

    return

def eval_4dfftcnn(model,val_set ='../../extract_raw_img',image_size=256,resume="",batch_size=16,num_workers=8,checkpoint="checkpoint",show_time=False):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss().to(device)

    model.load_state_dict(torch.load(os.path.join(checkpoint, resume)))

    dataloader_val = get_val_generate_4dfft(val_set,image_size,batch_size,num_workers)

    # train_losses, test_losses = [], []
    # import time
    test_loss = 0
    accuracy = 0
    model.eval()
    y_label = []
    y_pred = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_val):
            begin = time.time()
            # print(labels)
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            logps = model.forward(inputs)
            logps = logps.squeeze()
            # print(logps)
            logps_cpu = logps.cpu().numpy()
            y_pred.extend(logps_cpu.astype(np.float64))
            if show_time:
                print("Time : ", time.time() - begin)
            batch_loss = criterion(logps, labels)
            #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
            test_loss += batch_loss.item()
            #                     print("labels : ",labels)
            #                     print("logps  : ",logps)
            equals = labels == (logps > 0.5)
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)
            #                     print("equals   ",equals)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #                 train_losses.append(running_loss/len(trainloader))
    #             test_losses.append(test_loss/len(testloader))
    print(f"Test loss: {test_loss/len(dataloader_val):.3f}.. \n" +
          f"Test accuracy: {accuracy/len(dataloader_val):.3f}\n" +
          f"Test log_loss: {log_loss(y_label,y_pred,labels=np.array([0.,1.])):.3f}\n" +
          f"Test accuracy_score: {accuracy_score(y_label,y_pred_label):.3f}\n" +
          f"Test precision_score: {precision_score(y_label,y_pred_label):.3f}\n" +
          f"Test recall: {recall_score(y_label,y_pred_label):.3f}\n")
    print(classification_report(y_label,y_pred_label))

    return

if __name__ == "__main__":
    from pytorch_model.xception import xception
    model = xception(pretrained=False)
    eval_cnn(model,val_set ='../../../extract_raw_img_test',checkpoint="../../../model/xception/",resume="model_pytorch_4.pt")
    # eval_capsule(val_set ='../../../extract_raw_img_test',checkpoint="../../../model/capsule/",resume=6)