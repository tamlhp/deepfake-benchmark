import math
import torch
import torch.nn as nn
from pytorch_model.drn.drn import drn_c_26


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, classes, pretrained_drn=False,
            pretrained_model=None, use_torch_up=False):
        super(DRNSeg, self).__init__()

        model = drn_c_26(pretrained=pretrained_drn)
        self.base = nn.Sequential(*list(model.children())[:-2])
        if pretrained_model:
            self.load_pretrained(pretrained_model)

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)

        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

    def load_pretrained(self, pretrained_model):
        print("loading the pretrained drn model from %s" % pretrained_model)
        state_dict = torch.load(pretrained_model, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # filter out unnecessary keys
        pretrained_dict = state_dict['model']
        pretrained_dict = {k[5:]: v for k, v in pretrained_dict.items() if k.split('.')[0] == 'base'}

        # load the pretrained state dict
        self.base.load_state_dict(pretrained_dict)


class DRNSub(nn.Module):
    def __init__(self, num_classes, pretrained_model=None, fix_base=False):
        super(DRNSub, self).__init__()

        drnseg = DRNSeg(2)
        if pretrained_model:
            print("loading the pretrained drn model from %s" % pretrained_model)
            state_dict = torch.load(pretrained_model, map_location='cpu')
            drnseg.load_state_dict(state_dict['model'])

        self.base = drnseg.base
        if fix_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.Sigmoid()(x)
        return x


def load_classifier(model_path, gpu_id):
    if torch.cuda.is_available() and gpu_id != -1:
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    model = DRNSub(1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.device = device
    model.eval()
    return model

import argparse
import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
def classify_fake(model, img_path, no_crop=False,
                  model_file='utils/dlib_face_detector/mmod_human_face_detector.dat'):
    # Data preprocessing
    face = Image.open(img_path).size
    # if no_crop:
    #     face = Image.open(img_path).convert('RGB')
    # else:
    #     faces = face_detection(img_path, verbose=False, model_file=model_file)
    #     if len(faces) == 0:
    #         print("no face detected by dlib, exiting")
    #         sys.exit()
    #     face, box = faces[0]
    # face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(model.device)

    # Prediction
    with torch.no_grad():
        prob = model(face_tens.unsqueeze(0))[0].sigmoid().cpu().item()
    return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True, help="the model input")
    parser.add_argument(
        "--model_path", required=True, help="path to the drn model")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="do not use a face detector, instead run on the full input image")
    args = parser.parse_args()

    model = load_classifier(args.model_path, args.gpu_id)
    prob = classify_fake(model, args.input_path, args.no_crop)
    print("Probibility being modified by Photoshop FAL: {:.2f}%".format(prob*100))