import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('/home/dell/Downloads/69999.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Spectrum Image'), plt.xticks([]), plt.yticks([])
# plt.show()
plt.savefig('fourier.png', dpi=300, bbox_inches="tight")
