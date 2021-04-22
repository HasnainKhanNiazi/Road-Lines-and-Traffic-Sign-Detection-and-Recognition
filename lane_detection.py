# -*- coding: utf-8 -*-
"""
Created on Tue June 5 15:00:45 2020

@author: Hasnain Khan
"""

import cv2
import numpy as np

def Get_ROI_Vertices(image):
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    ROI_vertices = [
        (0, image_height),
        (image_width/2, image_height/2),
        (image_width, image_height)
    ]
    
    return ROI_vertices

def process_main(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    ROI_vertices = Get_ROI_Vertices(image)
    
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,np.array([ROI_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image,rho = 2,theta = np.pi/180 , threshold = 100,minLineLength = 100,maxLineGap = 100)
    image_with_lines = drow_the_lines(image, lines)
    return image_with_lines


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow("Masked Image",masked_image)
    return masked_image

def drow_the_lines(img, lines):
    #img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img




def read_image_or_Video():
    cap = cv2.VideoCapture('test2.mp4')
    
    while cap.isOpened():
        ret, image = cap.read()
        frame = process_main(image)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_image_or_Video()