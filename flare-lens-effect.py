#!/usr/bin/python2.7

import numpy as np
from imutils import paths
import cv2
import cv2 as cv
import argparse

#Define Canny function
ratio = 3
kernel_size = 3

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)

    return detected_edges


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
args = vars(ap.parse_args())

# loop over the input images
for imagePath in paths.list_images(args["images"]):
    img = cv2.imread(imagePath)

    # Scaling image to display it
    imgScale = 0.2  # W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    src = cv.resize(img, (int(newX), int(newY)))

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    aftercanny = CannyThreshold(40)

    kernel = np.ones((5, 5), np.uint8)
    afil = cv.dilate(aftercanny, kernel, iterations=1)



    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector(params)
    # Detect blobs.
    keypoints = detector.detect(afil)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(src, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)


