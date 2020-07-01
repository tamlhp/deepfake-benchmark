import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import torch.nn as nn
import argparse
from PIL import Image
import torchvision.transforms as transforms
import glob
import torch
# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--img_path', default="../../data/extract_raw_img_test/df/aahncigwte.mp4_0.jpg", help='path to image data ')
    parser.add_argument('--model_path', default="../../model/xception/model_pytorch_4.pt", help='path to model ')
    parser.add_argument('--gpu_id',type=int, default=-1, help='path to model ')
    parser.add_argument('--image_size',type=int, default=256, help='path to model ')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,resnext50, resnext ,gan,meso,xception')
    ## torch
    parser_capsule = subparsers.add_parser('capsule', help='Capsule')
    parser_drn = subparsers.add_parser('drn', help='DRN  ')
    parser_local_nn = subparsers.add_parser('local_nn', help='Local NN ')
    parser_self_attention = subparsers.add_parser('self_attention', help='Self Attention ')
    parser_resnext50 = subparsers.add_parser('resnext50', help='Resnext50 ')
    parser_resnext101 = subparsers.add_parser('resnext101', help='Resnext101 ')
    parser_myresnext = subparsers.add_parser('myresnext', help='My Resnext ')
    parser_mnasnet = subparsers.add_parser('mnasnet', help='mnasnet pytorch ')
    parser_xception_torch = subparsers.add_parser('xception_torch', help='Xception pytorch ')
    parser_xception2_torch = subparsers.add_parser('xception2_torch', help='Xception2 pytorch ')
    parser_dsp_fwa = subparsers.add_parser('dsp_fwa', help='DSP_SWA pytorch ')
    parser_meso = subparsers.add_parser('meso4_torch', help='Mesonet4')

    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')
    parser_efficient = subparsers.add_parser('efficient', help='Efficient Net')
    parser_efficient.add_argument("--type",type=str,required=False,default="0",help="Type efficient net 0-8")
    parser_efficientdual = subparsers.add_parser('efficientdual', help='Efficient Net')
    parser_efft = subparsers.add_parser('efft', help='Efficient Net fft')
    parser_efft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    parser_e4dfft = subparsers.add_parser('e4dfft', help='Efficient Net 4d fft')
    parser_e4dfft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    image_size = args.image_size
    transform = transforms.Compose([transforms.Resize((image_size,image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])

                                           ])
    transform_fft = transforms.Compose([transforms.ToTensor()])

    img = Image.open(args.img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    if model== "capsule":
        from pytorch_model.detect_torch import detect_capsule
        detect_capsule(img = img,gpu_id= gpu_id,model_path=args.model_path)
        exit(0)
        pass
    elif model == "drn" :
        from pytorch_model.drn.drn_seg import DRNSub
        model = DRNSub(1)
        pass
    elif model == "local_nn" :
        from pytorch_model.local_nn import local_nn
        model = local_nn()
    elif model == "self_attention":
        from pytorch_model.self_attention import self_attention
        model = self_attention()
    elif model == "resnext50":
        from pytorch_model.model_cnn_pytorch import resnext50
        model = resnext50(False)
    elif model == "resnext101":
        from pytorch_model.model_cnn_pytorch import resnext101
        model = resnext101(False)
    elif model == "myresnext":
        from pytorch_model.model_cnn_pytorch import MyResNetX
        model = MyResNetX()
    elif model == "mnasnet":
        from pytorch_model.model_cnn_pytorch import mnasnet
        model = mnasnet(False)
    elif model == "xception_torch":
        from pytorch_model.xception import xception
        model = xception(pretrained=False)
    elif model == "xception2_torch":
        from pytorch_model.xception import xception2
        model = xception2(pretrained=False)

    elif model == "meso4_torch":
        from pytorch_model.model_cnn_pytorch import mesonet
        model = mesonet(image_size=args.image_size)
    elif model == "dsp_fwa":
        from pytorch_model.DSP_FWA.models.classifier import SPPNet
        model = SPPNet(backbone=50, num_class=1)
    elif model == "siamese_torch":
        from pytorch_model.siamese import SiameseNetworkResnet
        model = SiameseNetworkResnet(length_embed = args.length_embed,pretrained=True)
    elif model == "efficient":
        from pytorch_model.efficientnet import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b'+args.type,num_classes=1)
        model = nn.Sequential(model,nn.Sigmoid())

    elif model == "efft":
        from pytorch_model.efficientnet import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=1)
        model = nn.Sequential(model, nn.Sigmoid())
    elif model == "e4dfft":
        from pytorch_model.efficientnet import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=4)
        model = nn.Sequential(model, nn.Sigmoid())
    elif model == "efficientdual":
        from pytorch_model.efficientnet import EfficientDual
        from pytorch_model.detect_torch import detect_dualcnn
        import numpy as np
        import  cv2
        model = EfficientDual()

        img = cv2.imread(args.img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        f = np.fft.fft2(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        fshift = np.fft.fftshift(f)
        fshift += 1e-8

        magnitude_spectrum = np.log(np.abs(fshift))
        # img = np.concatenate([img,magnitude_spectrum],axis=2)
        # img = np.transpose(img,(2,0,1))
        magnitude_spectrum = cv2.resize(magnitude_spectrum,(image_size,image_size))
        magnitude_spectrum = np.array([magnitude_spectrum])
        magnitude_spectrum = np.transpose(magnitude_spectrum, (1,2 , 0))
        PIL_img = Image.fromarray(img)
        # PIL_magnitude_spectrum = Image.fromarray(magnitude_spectrum)
        PIL_img = transform(PIL_img)
        magnitude_spectrum = transform_fft(magnitude_spectrum)
        magnitude_spectrum = magnitude_spectrum.unsqueeze(0)
        detect_dualcnn(model,PIL_img,magnitude_spectrum,model_path=args.model_path)
        exit(0)
    from pytorch_model.detect_torch import detect_cnn
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    detect_cnn(model, img)
    # for p in glob.glob("../../extract_raw_img/real/*.jpg"):
    #     img = Image.open(p)
    #     img = transform(img)
    #     img = img.unsqueeze(0)
    #     print(img)
        # detect_cnn(model,img)
