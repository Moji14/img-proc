#!/usr/bin/python2.7

from matplotlib import pyplot as plt
from imutils import paths
import cv2
import cv2 as cv
import argparse

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
        imgScale = 0.1  # W / width
        newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
        img = cv.resize(img, (int(newX), int(newY)))

        #Convert to gray scale
        bitgrayimg = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #bitgrayimg = cv.bitwise_not(bitgrayimg)

        #Equalization
        #bitgrayimg = cv2.equalizeHist(bitgrayimg)

        bitgrayimg = bitgrayimg

        #Threshold calculation
        ret, th1 = cv.threshold(bitgrayimg, 240, 256, cv.THRESH_BINARY)
        th2 = cv.adaptiveThreshold(bitgrayimg, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 11, 2)
        th3 = cv.adaptiveThreshold(bitgrayimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 11, 2)
        # Display preparing
        titles = ['Original Image', 'Global Thresholding (v = 240)','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [bitgrayimg, th1, th2, th3]
        plt.figure()
        for i in xrange(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        #Display results
        #print 'prepping plot done'
        #cv.imshow(imagePath, bitgrayimg)
        plt.show()
        print 'after ploting test'