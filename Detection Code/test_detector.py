# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:00:45 2020

@author: Hasnain Khan
"""
import cv2
import dlib
import glob


def detect_sign():
    j = 0
    detector = dlib.simple_object_detector("detector.svm")
    
    object_detector = dlib.image_window()
    object_detector.set_image(detector)    
    
    for image in glob.glob("C:/Users/MoHSiN/Desktop/Hasnain/Project/Detection Code/Detection_Dataset/Test_Dataset/Dataset/*g"):
        #print(str(image))
        print(j)
        j = j + 1
        img = cv2.imread(image,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        win = dlib.image_window()
    
        detected_sign = detector(img)
        
        for (i, rect) in enumerate(detected_sign):
            x1, y1, x2, y2, w, h = rect.left(), rect.top(), rect.right() + \
                1, rect.bottom() + 1, rect.width(), rect.height()

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 128, 0), 2) # First Method to draw rectangle
            #win.add_overlay(detected_sign) # Second Method to draw rectangle
       
        image_name = "cropped_" + str(j) + ".jpg"
        roi_color = img[y1:y1 + h, x1:x1 + w]
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_name, roi_color)
        win.set_image(img)
        
        #input("Press Enter to continue...")


if __name__ == "__main__":
    detect_sign()
