import torch
import numpy as np
from torch.autograd import Variable
from pytorch_model.capsule_pytorch.model import VggExtractor,CapsuleNet,CapsuleLoss
import torch.nn as nn


def detect_capsule(img,gpu_id=-1,model_path="checkpoint"):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    vgg_ext = VggExtractor().to(device)
    capnet = CapsuleNet(2,gpu_id)
    capnet = capnet.to(device)

    capnet.load_state_dict(torch.load(model_path))

    input_v = Variable(img)

    x = vgg_ext(input_v)
    classes, class_ = capnet(x, random=False)
    output_dis = class_.data.detach().numpy()
    output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

    for i in range(output_dis.shape[0]):
        if output_dis[i,1] >= output_dis[i,0]:
            output_pred[i] = 1.0
        else:
            output_pred[i] = 0.0

    print(output_pred)
    return output_pred

def detect_cnn(model,img):

    logps = model.forward(img)
    logps = logps.squeeze()
    logps_cpu = logps.detach().numpy()
    print(logps_cpu)
    pred_label = (logps_cpu > 0.5)

    print(pred_label)
    return pred_label

def detect_dualcnn(model,img,img_fft,model_path="checkpoint"):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logps = model.forward(img.float().to(device),img_fft.float().to(device))
    logps = logps.squeeze()
    logps_cpu = logps.detach().numpy()
    pred_label = (logps_cpu > 0.5)
    print(pred_label)

    return pred_label
