import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image
def dwt(x):

    x01 = x[ :, 0::2, :] / 2
    x02 = x[ :, 1::2, :] / 2
    x1 = x01[:, :, 0::2] /2
    x2 = x02[ :, :, 0::2] /2
    x3 = x01[:, :, 1::2] /2
    x4 = x02[:, :, 1::2]/ 2
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

def iwt(x_LL, x_HL, x_LH, x_HH):

    x1 = x_LL
    x2 = x_HL
    x3 = x_LH
    x4 = x_HH
    in_channel, in_height, in_width = x_LL.size()
    out_channel, out_height, out_width = in_channel, 2 * in_height, 2 * in_width

    h = torch.zeros([out_channel, out_height, out_width]).float()

    h[:, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[ :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

def test_wave(args):
    noisy_path = args.img
    # clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    # image_file = noisy_path[2]
    image_file = noisy_path
    image_noise = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
    # gt = transforms.ToTensor()(Image.open(clean_path[2]).convert('RGB'))
    x_LL, x_HL, x_LH, x_HH = dwt(image_noise)
    # x_LL2, x_HL2, x_LH2, x_HH2 = dwt(x_HL)
    # x_LL3, x_HL3, x_LH3, x_HH3 = dwt(x_LL2)
    # x_LL4, x_HL4, x_LH4, x_HH4 = dwt(x_LL3)
    # x_LL5, x_HL5, x_LH5, x_HH5 = dwt(x_LL4)


    x_LL_gt, x_HL_gt, x_LH_gt, x_HH_gt = dwt(image_noise)
    # x_LL_gt2, x_HL_gt2, x_LH_gt2, x_HH_gt2 = dwt(x_HL_gt)
    # x_LL_gt3, x_HL_gt3, x_LH_gt3, x_HH_gt3 = dwt(x_LL_gt2)
    # x_LL_gt4, x_HL_gt4, x_LH_gt4, x_HH_gt4 = dwt(x_LL_gt3)
    # x_LL, x_HL, x_LH, x_HH = dwt(x_LL)
    # x_LL, x_HL, x_LH, x_HH = dwt(x_LL)
    # x_LL, x_HL, x_LH, x_HH = dwt(x_LL)

    trans = transforms.ToPILImage()
    # denoise_LL3 = iwt(x_LL_gt4,x_HL_gt4, x_LH_gt4, x_HH_gt4)
    # denoise_LL2 = iwt(denoise_LL3,x_HL_gt3, x_LH_gt3, x_HH_gt3)
    # denoise_LL1 = iwt(denoise_LL2,x_HL_gt2, x_LH_gt2, x_HH_gt2)
    # denoise_LL = iwt(x_LL,x_HL_gt, x_LH_gt, x_HH_gt)
    # psnr_t = calculate_psnr(denoise_LL, gt)
    # ssim_t = calculate_ssim(denoise_LL, gt)
    # print("   UP   :  PSNR : ", str(psnr_t), " :  SSIM : ", str(ssim_t))
    plt.figure(figsize=(9, 9))
    plt.subplot(2, 2, 1)
    plt.imshow(np.array(trans(x_LL_gt)))
    plt.title("LL" , fontsize=22)
    plt.subplot(2, 2, 2)
    plt.imshow(np.array(trans(x_HL_gt)))
    plt.title("HL" , fontsize=22)
    plt.subplot(2, 2, 3)
    plt.imshow(np.array(trans(x_LH_gt)))
    plt.title("LH" , fontsize=22)
    plt.subplot(2, 2, 4)
    plt.imshow(np.array(trans(x_HH_gt)))
    plt.title("HH" , fontsize=22)
    # plt.show()
    plt.savefig('wavelet_exp.png', dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--img','-i', default='/home/dell/Downloads/69999.png', help='path to noise image file')

    args = parser.parse_args()
    #

    test_wave(args)
