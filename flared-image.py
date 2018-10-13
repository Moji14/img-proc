#!/usr/bin/python2.7
# import the necessary packages
import argparse
import cv2
import cv2 as cv
import numpy as np
import copy
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--radius", type = int,
	help = "radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# loop over the input images
for imagePath in paths.list_images(args["image"]):
    #Load Image
    img = cv2.imread(imagePath)

    # Scaling image to display it
    imgScale = 0.2  # W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))

    #Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, th = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)

    #dst = cv.addWeighted(img, 0.7, th, 0.3, 0)


    b, g, r = cv2.split(img)

    new_r = r - th

    new_g = g - th

    img = cv2.merge((b, new_g, new_r))

    # display the results of our newly improved method
    cv2.imshow("Robust", img)
    cv2.waitKey(0)
