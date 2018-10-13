#!/usr/bin/python2.7

from __future__ import print_function
import cv2
import cv2 as cv
import argparse
from imutils import paths

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
args = vars(ap.parse_args())

for imagePath in paths.list_images(args["images"]):

    img = cv2.imread(imagePath)

    # Scaling image to display it
    imgScale = 0.2  # W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv.resize(img, (int(newX), int(newY)))

    src = img

    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)
    cv.waitKey()