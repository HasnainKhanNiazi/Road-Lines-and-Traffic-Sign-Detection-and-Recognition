# -*- coding: utf-8 -*-
"""
Created on Sun May 17 05:21:51 2020

@author: Hasnain Khan
"""

import os
import cv2
import re
from skimage.feature import hog

def data_preprocess(image_path , annotation_path):
    print("Data Preprocessing")
    features = []
    labels = []
    with open(annotation_path) as file_in:
        for line in file_in:
            # line is the line taht we are reading one by one (1st, 2nd , 3rd, etc)        
            splitted_line = re.split(";",line) # splitting line for getting information
            image_name = splitted_line[0]
            label = splitted_line[7]
            for root, dirs, files in os.walk(image_path):
                if image_name in files:
                    image = os.path.join(root, image_name)
                    img = cv2.imread(image)
                    resized_image = cv2.resize(img,(100,100))
                    fd, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(10, 10),
                                        cells_per_block=(2, 2), visualize=True, block_norm="L2")
                    features.append(fd)
                    labels.append(label)
                else:
                    print("Image not found" , image)
    
    return features,labels

