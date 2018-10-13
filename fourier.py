#!/usr/bin/python2.7

import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
                help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# loop over the input images
for imagePath in paths.list_images(args["images"]):
        img = cv2.imread(imagePath)

        # Scaling image to display it
        imgScale = 0.2  # W / width
        newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
        newimg = cv2.resize(img, (int(newX), int(newY)))

        #Convert to gray
        gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

        #DFT calculation
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        plt.subplot(121), plt.imshow(img)
        plt.title(imagePath), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='inferno')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
