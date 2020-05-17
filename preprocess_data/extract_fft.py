# https://github.com/cc-hpc-itwm/DeepFakeDetection

import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import griddata


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

number_iter = 1000

psd1D_total = np.zeros([number_iter, N])
label_total = np.zeros([number_iter])
# psd1D_org_mean = np.zeros(N)
# psd1D_org_std = np.zeros(N)

cont = 0


# fake data
rootdir = 'dataset_celebA/'

for filename in glob.glob(rootdir + "*.jpg"):


    psd1D_total[cont, :] = interpolated
    label_total[cont] = 1
    cont += 1

    if cont == number_iter:
        break

# for x in range(N):
#     psd1D_org_mean[x] = np.mean(psd1D_total[:, x])
#     psd1D_org_std[x] = np.std(psd1D_total[:, x])

## real data
psd1D_total2 = np.zeros([number_iter, N])
label_total2 = np.zeros([number_iter])
# psd1D_org_mean2 = np.zeros(N)
# psd1D_org_std2 = np.zeros(N)

cont = 0
rootdir2 = '/home/duralllopez/DATASETS/celebA/img_align_celeba/'

for filename in glob.glob(rootdir2 + "*.jpg"):
    img = cv2.imread(filename, 0)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(magnitude_spectrum)

    points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
    xi = np.linspace(0, N, num=N)  # coordinates for interpolation

    interpolated = griddata(points, psd1D, xi, method='cubic')

    interpolated /= interpolated[0]

    psd1D_total2[cont, :] = interpolated
    label_total2[cont] = 0
    cont += 1

    if cont == number_iter:
        break

# for x in range(N):
#     psd1D_org_mean2[x] = np.mean(psd1D_total2[:, x])
#     psd1D_org_std2[x] = np.std(psd1D_total2[:, x])

# y.append(psd1D_org_mean)
# y.append(psd1D_org_mean2)
# error.append(psd1D_org_std)
# error.append(psd1D_org_std2)

psd1D_total_final = np.concatenate((psd1D_total, psd1D_total2), axis=0)
label_total_final = np.concatenate((label_total, label_total2), axis=0)

data["data"] = psd1D_total_final
data["label"] = label_total_final

output = open('train.pkl', 'wb')
pickle.dump(data, output)
output.close()

print("DATA Saved")

if __name__ == "__main__":
    pass