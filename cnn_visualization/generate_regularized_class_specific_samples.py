"""
Created on Tues Mar 10 08:13:15 2020
@author: Alex Stoken - https://github.com/alexstoken

Last tested with torchvision 0.5.0 with image and model on cpu
"""
import os
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.optim import SGD
from torch.autograd import Variable
from torchvision import models

from cnn_visualization.misc_functions import recreate_image, save_image
import argparse
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class RegularizedClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent. Uses Gaussian blur, weight decay, and clipping. 
    """

    def __init__(self, model, target_class,image_size):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model.cuda() if use_cuda else model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (image_size, image_size, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists(f'generated/class_{self.target_class}'):
            os.makedirs(f'generated/class_{self.target_class}')
        self.device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    def generate(self, iterations=150, blur_freq=4, blur_rad=1, wd=0.0001, clipping_value=0.1):
        """Generates class specific image with enhancements to improve image quality. 
        See https://arxiv.org/abs/1506.06579 for details on each argument's effect on output quality. 
        

        Play around with combinations of arguments. Besides the defaults, this combination has produced good images:
        blur_freq=6, blur_rad=0.8, wd = 0.05

        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})
            blur_freq {int} -- Frequency of Gaussian blur effect, in iterations (default: {6})
            blur_rad {float} -- Radius for gaussian blur, passed to PIL.ImageFilter.GaussianBlur() (default: {0.8})
            wd {float} -- Weight decay value for Stochastic Gradient Ascent (default: {0.05})
            clipping_value {None or float} -- Value for gradient clipping (default: {0.1})
        
        Returns:
            np.ndarray -- Final maximally activated class image
        """
        initial_learning_rate = 6
        for i in range(1, iterations):
            # Process image and return variable

            #implement gaussian blurring every ith iteration
            #to improve output
            if i % blur_freq == 0:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False, blur_rad)
            else:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False)

            if use_cuda:
                self.processed_image = self.processed_image.cuda()

            # Define optimizer for the image - use weight decay to add regularization
            # in SGD, wd = 2 * L2 regularization (https://bbabenko.github.io/weight-decay/)
            optimizer = SGD([self.processed_image],
                            lr=initial_learning_rate, weight_decay=wd)
            # Forward
            output = self.model(self.processed_image).cpu()
            # Target specific class
            class_loss = -output[0, self.target_class]

            if i in np.linspace(0, iterations, 10, dtype=int):
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.cpu().numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()

            if clipping_value:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), clipping_value)
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image.cpu())

            if i in np.linspace(0, iterations, 10, dtype=int):
                # Save image
                im_path = f'generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
                save_image(self.created_image, im_path)

        #save final image
        im_path = f'generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
        save_image(self.created_image, im_path)

        #write file with regularization details
        with open(f'generated/class_{self.target_class}/run_details.txt', 'w') as f:
            f.write(f'Iterations: {iterations}\n')
            f.write(f'Blur freq: {blur_freq}\n')
            f.write(f'Blur radius: {blur_rad}\n')
            f.write(f'Weight decay: {wd}\n')
            f.write(f'Clip value: {clipping_value}\n')

        #rename folder path with regularization details for easy access
        os.rename(f'generated/class_{self.target_class}',
                  f'generated/class_{self.target_class}_blurfreq_{blur_freq}_blurrad_{blur_rad}_wd{wd}')
        return self.processed_image


def preprocess_and_blur_image(pil_im, resize_im=True, blur_rad=None,image_size=256):
    """
        Processes image with optional Gaussian blur for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
        blur_rad (int): Pixel radius for Gaussian blurring (default = None)
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print(
                "could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im.thumbnail((image_size, image_size))

    #add gaussin blur to image
    if blur_rad:
        pil_im = pil_im.filter(ImageFilter.GaussianBlur(blur_rad))

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    if use_cuda:
        im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
    else:
        im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--model_path', default="../../model/xception/model_pytorch_4.pt", help='path to model ')
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
    gpu_id = 0 if int(args.gpu_id) >= 0 else -1
    image_size = args.image_size
    iterations = args.iterations
    if model == "capsule":
        exit(0)
        pass
    elif model == "drn":
        from pytorch_model.drn.drn_seg import DRNSub

        model = DRNSub(1)
        pass
    elif model == "local_nn":
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

        model = SiameseNetworkResnet(length_embed=args.length_embed, pretrained=True)
    elif model == "efficient":
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1)
        model = nn.Sequential(model, nn.Sigmoid())

    elif model == "efft":
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1, in_channels=1)
        model = nn.Sequential(model, nn.Sigmoid())
    elif model == "e4dfft":
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1, in_channels=4)
        model = nn.Sequential(model, nn.Sigmoid())
    elif model == "efficientdual":
        pass

    # from pytorch_model.xception import xception

    # model = xception(pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    print("Load xong ... ")
    model.eval()

    csig = RegularizedClassSpecificImageGeneration(model, target_class,image_size)
    csig.generate(iterations = iterations)
