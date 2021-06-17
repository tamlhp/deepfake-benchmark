"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import SGD

from cnn_visualization.misc_functions import preprocess_image, recreate_image, save_image
import argparse
import torch.nn as nn


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class,image_size):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.image_size = image_size
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (image_size, image_size, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('generated/class_'+str(self.target_class)):
            os.makedirs('generated/class_'+str(self.target_class))
        print("init xong ... ")
        self.device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    def generate(self, iterations=150):
        """Generates class specific image

        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})

        Returns:
            np.ndarray -- Final maximally activated class image
        """
        print("bat dau generate xong ... ")
        initial_learning_rate = 200
        for i in range(1, iterations):
            print(i)
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)

            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image.to(self.device))
            # Target specific class
            print(output)
            class_loss = -output[0, self.target_class]

            if i % 1 == 0 or i == iterations-1:
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.cpu().data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            print(self.created_image.size)
            if i % 1 == 0 or i == iterations-1:
                # Save image
                initial_learning_rate /=2
                im_path = 'generated/class_'+str(self.target_class)+'/c_'+str(self.target_class)+'_'+'iter_'+str(i)+'.png'
                save_image(self.created_image, im_path)

        return self.processed_image

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--model_path', default="../../../model/xception/model_pytorch_4.pt", help='path to model ')
    parser.add_argument('--gpu_id',type=int, default=-1, help='path to model ')
    parser.add_argument('--image_size',type=int, default=256, help='path to model ')
    parser.add_argument('--iterations',type=int, default=256, help='iterations random number')

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

    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')
    parser_efficient = subparsers.add_parser('efficient', help='Efficient Net')
    parser_efficient.add_argument("--type",type=str,required=False,default="0",help="Type efficient net 0-8")
    parser_efficientdual = subparsers.add_parser('efficientdual', help='Efficient Net')
    parser_efft = subparsers.add_parser('efft', help='Efficient Net fft')
    parser_efft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    parser_e4dfft = subparsers.add_parser('e4dfft', help='Efficient Net 4d fft')
    parser_e4dfft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    return parser.parse_args()
if __name__ == '__main__':
    target_class = 0  # Flamingo
    # pretrained_model = models.alexnet(pretrained=True)
    args = parse_args()
    print(args)
    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    image_size = args.image_size
    iterations= args.iterations
    if model== "capsule":
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
        pass


    from pytorch_model.xception import xception

    model = xception(pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path,map_location=torch.device('cpu')))
    print("Load xong ... ")
    model.eval()
    csig = ClassSpecificImageGeneration(model, target_class,image_size)
    csig.generate(iterations = iterations)
