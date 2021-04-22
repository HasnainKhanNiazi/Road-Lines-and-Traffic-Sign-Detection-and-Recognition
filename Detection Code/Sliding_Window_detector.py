# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:00:45 2020

@author: Hasnain Khan
"""

import cv2
from helpers import pyramid
from helpers import sliding_window
import joblib
from skimage.feature import hog
import time

def sliding_window_function():
    image = cv2.imread("/home/hasnain/Desktop/Computer Vision Workspace/AIP_Project/nopassing.png")
    # image = cv2.resize(image,(32,32))
    (winW, winH) = (32, 32)

    clf = joblib.load("finalized_model.sav")
    # clf.detectM
    # loop over the image pyramid
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            fd, hog_image = hog(window, orientations=9, pixels_per_cell=(10, 10),
                                        cells_per_block=(2, 2), visualize=True, block_norm="L2")
            #fd = fd.reshape(1,-1)
            results = clf.predict(fd)
            print(results)

            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(5)
            time.sleep(0.025)