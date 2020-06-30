import cv2
import numpy as np
import os, pickle, argparse
from feature_model.spectrum import radialProfile
import glob
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import griddata
from PIL import ImageEnhance,Image
import random

data= {}
epsilon = 1e-8
N = 300
y = []
error = []

def main(input_real,input_fake,number_iter,output_path):
    random.seed(0)
    psd1D_total = np.zeros([number_iter, N])
    label_total = np.zeros([number_iter])
    cont = 0

    # real data
    # rootdir = '/hdd/tam/extend_data/image/test/0_real'
    rootdirs = [input_real,input_fake]
    for subdir, dirs, files in os.walk(rootdirs[0]):
        #     print(files)
        #     exit(0)
        random.shuffle(files)
        for file in files:
            #         print(file)
            filename = os.path.join(subdir, file)

            img = cv2.imread(filename, 0)

            img = img.astype("uint8")
            contrast = ImageEnhance.Contrast(Image.fromarray(img))
            img = contrast.enhance(1.0)
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(2.0)
            img = np.array(img, dtype='float64')

            # we crop the center
            h = int(img.shape[0] / 3)
            w = int(img.shape[1] / 3)
            img = img[h:-h, w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += epsilon
            try:
                magnitude_spectrum = 20 * np.log(np.abs(fshift))
                psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

                # Calculate the azimuthally averaged 1D power spectrum
                points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
                xi = np.linspace(0, N, num=N)  # coordinates for interpolation

                interpolated = griddata(points, psd1D, xi, method='cubic')
                interpolated /= interpolated[0]

                psd1D_total[cont, :] = interpolated
                label_total[cont] = 0
                cont += 1
            #             print(cont)
            except:
                print(file)
            if cont == number_iter:
                break
        if cont == number_iter:
            break

    psd1D_total2 = np.zeros([number_iter, N])
    label_total2 = np.zeros([number_iter])

    cont = 0

    for subdir, dirs, files in os.walk(rootdirs[1]):

        random.shuffle(files)
        #     print(files)
        #     break
        for file in files:

            filename = os.path.join(subdir, file)
            parts = filename.split("/")

            img = cv2.imread(filename, 0)

            img = img.astype("uint8")
            contrast = ImageEnhance.Contrast(Image.fromarray(img))
            img = contrast.enhance(1.0)
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(2.0)
            img = np.array(img, dtype='float64')

            # we crop the center
            h = int(img.shape[0] / 3)
            w = int(img.shape[1] / 3)
            img = img[h:-h, w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += epsilon

            try:
                magnitude_spectrum = 20 * np.log(np.abs(fshift))

                # Calculate the azimuthally averaged 1D power spectrum
                psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

                points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
                xi = np.linspace(0, N, num=N)  # coordinates for interpolation

                interpolated = griddata(points, psd1D, xi, method='cubic')
                interpolated /= interpolated[0]

                psd1D_total2[cont, :] = interpolated
                label_total2[cont] = 1
                cont += 1
            #             print(cont)
            except:
                print(file)
            if cont == number_iter:
                break
        if cont == number_iter:
            break
    psd1D_total_final = np.concatenate((psd1D_total,psd1D_total2), axis=0)
    label_total_final = np.concatenate((label_total,label_total2), axis=0)

    data["data"] = psd1D_total_final
    data["label"] = label_total_final

    output = open(output_path, 'wb')
    pickle.dump(data, output)
    output.close()

    print("DATA Saved")


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-ir', '--input_real', dest='input_real',default='',
                        help='Path to input image or folder containting multiple images.')
    parser.add_argument('-if', '--input_fake', dest='input_fake',default='',
                        help='Path to input image or folder containting multiple images.')
    parser.add_argument('-o', '--output', dest='output', help='Path to save outputs.',
                        default='./output')
    parser.add_argument('-n', '--number_iter', default=100,help='number image process')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args_in = parse_args()
    main(args_in.input_real,args_in.input_fake, args_in.number_iter,args_in.ouput)
