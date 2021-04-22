# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:00:45 2020

@author: Hasnain Khan
"""

import os
import sys
import glob

import dlib


faces_folder = "C:/Users/Hasnain/Desktop/Dataset_new"


options = dlib.simple_object_detector_training_options()
options.C = 5
options.num_threads = 4
options.be_verbose = True

training_xml_path = os.path.join(faces_folder, "C:/Users/Hasnain/Desktop/newmydataset.xml")
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)
