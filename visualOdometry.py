# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:04:08 2020

@author: Praveen
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from FeatureDetectionAndMatching import featureDetectionMatching
from RANSAC import outlierRejection
from ReadCameraModel import ReadCameraModel
from EstimateEssentialMatrix import essentialMatrix
from CameraPose import obtainCameraPose
from UniqueCameraPose import linearTriangulation, obtainUniqueCameraPose
# from plotVO import DrawCameras

H = np.identity(4)
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./model')
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
p0 = np.array([0, 0, 0, 1]).T

for i in range(23, 3872):
    frame_1 = cv2.imread(r'C:/Users/Praveen/ENPM 673 Project 5/dataset/data/' + str(i) +'.png', 0)
    frame_2 = cv2.imread(r'C:/Users/Praveen/ENPM 673 Project 5/dataset/data/' + str(i + 1) +'.png', 0)
    frame_1 = frame_1[220:650, 0:1280]
    frame_2 = frame_2[220:650, 0:1280]
    srcFeatures, dstFeatures = featureDetectionMatching(frame_1, frame_2)
    F, inlierSrc, inlierDst = outlierRejection(srcFeatures, dstFeatures)
    E = essentialMatrix(F, K)
    cameraCenter, rotationMatrix = obtainCameraPose(E)

    X = []
    for i in range(4):
        X.append(linearTriangulation(np.zeros((3, 1)), cameraCenter[i], np.identity(3), rotationMatrix[i], srcFeatures, dstFeatures, K))

    cameraCenter, rotationMatrix, X = obtainUniqueCameraPose(cameraCenter, rotationMatrix, X)
    cameraCenter = cameraCenter.reshape((3, 1))
    
    H_current = np.hstack([rotationMatrix, cameraCenter])
    H_current_new = np.vstack([H_current, np.array([[0, 0, 0, 1]]).reshape((1, 4))])
    H = np.matmul(H, H_current_new)
    projection = np.matmul(H, p0)
    print(projection)
    x = projection[0]
    z = projection[2]
    plt.scatter(x, -z, color = 'r')
    if i % 50 == 0:
        plt.pause(1)
    else:
        plt.pause(0.1)
    





