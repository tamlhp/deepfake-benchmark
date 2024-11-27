import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--img_path', default="../../data/extract_raw_img_test/df/aahncigwte.mp4_0.jpg", help='Path to image data')
    parser.add_argument('--model_path', default="../../model/xception/model_pytorch_4.pt", help='Path to model')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID for CUDA')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for preprocessing')

    subparsers = parser.add_subparsers(dest="model", help='Choose a model from: capsule, drn, resnext50, resnext, gan, meso, xception')
    models = [
        'capsule', 'drn', 'local_nn', 'self_attention', 'resnext50', 'resnext101',
        'myresnext', 'mnasnet', 'xception_torch', 'xception2_torch', 'dsp_fwa', 
        'meso4_torch', 'efficient', 'efficientdual', 'efft', 'e4dfft'
    ]
    for model in models:
        subparsers.add_parser(model, help=model.capitalize())

    return parser.parse_args()

def load_image(img_path, image_size):
    try:
        img = Image.open(img_path)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img).unsqueeze(0)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)

def load_model(model_name, model_path):
    try:
        if model_name == "capsule":
            from pytorch_model.detect_torch import detect_capsule
            return detect_capsule
        elif model_name == "drn":
            from pytorch_model.drn.drn_seg import DRNSub
            model = DRNSub(1)
        elif model_name == "local_nn":
            from pytorch_model.local_nn import local_nn
            model = local_nn()
        elif model_name == "self_attention":
            from pytorch_model.self_attention import self_attention
            model = self_attention()
        elif model_name in ["resnext50", "resnext101", "myresnext", "mnasnet"]:
            from pytorch_model.model_cnn_pytorch import resnext50, resnext101, MyResNetX, mnasnet
            model = {"resnext50": resnext50, "resnext101": resnext101, "myresnext": MyResNetX, "mnasnet": mnasnet}[model_name](False)
        elif model_name == "xception_torch":
            from pytorch_model.xception import xception
            model = xception(pretrained=False)
        elif model_name == "xception2_torch":
            from pytorch_model.xception import xception2
            model = xception2(pretrained=False)
        elif model_name == "meso4_torch":
            from pytorch_model.model_cnn_pytorch import mesonet
            model = mesonet(image_size=image_size)
        elif model_name == "dsp_fwa":
            from pytorch_model.DSP_FWA.models.classifier import SPPNet
            model = SPPNet(backbone=50, num_class=1)
        elif model_name == "efficient":
            from pytorch_model.efficientnet import EfficientNet
            model = EfficientNet.from_pretrained(f'efficientnet-b{args.type}', num_classes=1)
            model = nn.Sequential(model, nn.Sigmoid())
        elif model_name == "efft":
            from pytorch_model.efficientnet import EfficientNet
            model = EfficientNet.from_pretrained(f'efficientnet-b{args.type}', num_classes=1, in_channels=1)
            model = nn.Sequential(model, nn.Sigmoid())
        elif model_name == "e4dfft":
            from pytorch_model.efficientnet import EfficientNet
            model = EfficientNet.from_pretrained(f'efficientnet-b{args.type}', num_classes=1, in_channels=4)
            model = nn.Sequential(model, nn.Sigmoid())
        elif model_name == "efficientdual":
            from pytorch_model.efficientnet import EfficientDual
            from pytorch_model.detect_torch import detect_dualcnn
            return EfficientDual, detect_dualcnn
        else:
            raise ValueError("Invalid model name")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    gpu_id = 0 if int(args.gpu_id) >= 0 else -1
    img = load_image(args.img_path, args.image_size)
    model = load_model(args.model, args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() and gpu_id != -1 else "cpu")
    model = model.to(device)

    if hasattr(model, 'load_state_dict'):
        try:
            model.load_state_dict(torch.load(args.model_path))
            model.eval()
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            exit(1)

    if args.model == "capsule":
        model(img, gpu_id, args.model_path)
    elif args.model == "efficientdual":
        img_cv = cv2.imread(args.img_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        f = np.fft.fft2(cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY))
        fshift = np.fft.fftshift(f) + 1e-8
        magnitude_spectrum = np.log(np.abs(fshift))
        magnitude_spectrum = cv2.resize(magnitude_spectrum, (args.image_size, args.image_size))
        magnitude_spectrum = torch.tensor(magnitude_spectrum).unsqueeze(0).unsqueeze(0).to(device)

        img_tensor = img.to(device)
        detect_dualcnn(model, img_tensor, magnitude_spectrum, model_path=args.model_path)
    else:
        detect_cnn(model, img.to(device))

if __name__ == "__main__":
    main()
