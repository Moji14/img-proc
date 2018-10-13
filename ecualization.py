#!/usr/bin/python2.7

import cv2
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

    #convert to grayscale
    gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

    # Image equalization
    equ = cv2.equalizeHist(gray)
    #res = np.hstack((gray, equ))  # stacking images side-by-side


    #Display images and histograms
    plt.subplot(2, 2, 1), plt.imshow(gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(equ, cmap='gray')
    plt.title('Equalized'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.hist(gray.ravel(), 256, [0, 256])
    plt.title('Hitogram gray'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.hist(equ.ravel(),256,[0,256])
    plt.title('Histogram equialized'), plt.xticks([]), plt.yticks([])
    plt.show()
