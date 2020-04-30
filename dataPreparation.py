# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:07:56 2020

@author: Praveen
"""

import cv2
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import glob

def dataPrep(readPath, writePath):
    imagePath = readPath
    imageName = [image for image in glob.glob(imagePath)]
    imageName.sort()
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./model')
    i = 0
    for image in imageName:
        distorted_bayerImage = cv2.imread(image, 0)
        distortedImage = cv2.cvtColor(distorted_bayerImage, cv2.COLOR_BayerGR2BGR)
        undistortedImage = UndistortImage(distortedImage, LUT)
        undistortedImage = cv2.cvtColor(undistortedImage, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(writePath + "/data/" + str(i) + ".png", undistortedImage)
        i = i + 1
    
def main():
    readPath = "dataset/stereo/centre/*.png"
    writePath = r"C:/Users/Praveen/ENPM Project 5/dataset"
    dataPrep(readPath, writePath)
    
if __name__ == '__main__':
    main()