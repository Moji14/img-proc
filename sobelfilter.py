#!/usr/bin/python2.7
import numpy as np
from matplotlib import pyplot as plt
from imutils import paths
import cv2
import cv2 as cv
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
args = vars(ap.parse_args())

# loop over the input images
for imagePath in paths.list_images(args["images"]):
    img = cv2.imread(imagePath)

    # Scaling image to display it
    imgScale = 0.2  # W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))

    #Change to Grayscale
    img = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Equalization
    img = cv2.equalizeHist(img)


    # Output dtype = cv2.CV_8U
    sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)


    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobel_8u, cmap='gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobel_8u, cmap='gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    plt.show()
