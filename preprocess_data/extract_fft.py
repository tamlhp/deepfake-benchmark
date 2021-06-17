# https://github.com/cc-hpc-itwm/DeepFakeDetection

import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import griddata
import argparse
import traceback


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
def get_interpolated(img):

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = azimuthalAverage(magnitude_spectrum)

    # Calculate the azimuthally averaged 1D power spectrum
    points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
    xi = np.linspace(0, N, num=N)  # coordinates for interpolation

    interpolated = griddata(points, psd1D, xi, method='cubic')
    interpolated /= interpolated[0]
    return interpolated

data = {}
epsilon = 1e-8
N = 80
y = []
error = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--in_train', default="data/train/", help='path to train data ')
    parser.add_argument('--in_val', default="data/test/", help='path to test data ')
    parser.add_argument('--out_train', type=str, default="train_feature.pkl", help='out_train')
    parser.add_argument('--out_val', type=str, default="val_feature.pkl", help='out_val')

    args = parser.parse_args()
    features = []
    try:
        for i in glob.glob(args.in_train):
            print(i)
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feature = get_interpolated(img)
            features.append(feature)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        # pass
    output = open(args.out_train, 'wb')
    pickle.dump(features, output)
    output.close()

    features = []
    try:
        for i in glob.glob(args.in_val):
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feature = get_interpolated(img)
            features.append(feature)
            # pass
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
    output = open(args.out_val, 'wb')
    pickle.dump(features, output)
    output.close()