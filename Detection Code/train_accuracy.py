# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:00:45 2020

@author: Hasnain Khan
"""

import os
import sys
import glob

import dlib

images_folder = "C:/Users/Hasnain/Desktop/Dataset_new"
training_xml_path = os.path.join(images_folder, "C:/Users/Hasnain/Desktop/newmydataset.xml")

accuracy = dlib.test_simple_object_detector(training_xml_path, "detector.svm")
 
print("Training Accuracy : " , accuracy)   # 91%