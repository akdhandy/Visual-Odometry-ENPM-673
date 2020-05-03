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

frame_1 = cv2.imread(r'C:\Users\Praveen\ENPM 673 Project 5\dataset\data\22.png', 0)
frame_2 = cv2.imread(r'C:\Users\Praveen\ENPM 673 Project 5\dataset\data\23.png', 0)
srcFeatures, dstFeatures = featureDetectionMatching(frame_1, frame_2)
F, inlierSrc, inlierDst = outlierRejection(srcFeatures, dstFeatures)


rows1 = frame_1.shape[0]
cols1 = frame_1.shape[1]
rows2 = frame_2.shape[0]
cols2 = frame_2.shape[1]
out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
out[:rows1,:cols1,:] = np.dstack([frame_1, frame_1, frame_1])
out[:rows2,cols1:cols1+cols2,:] = np.dstack([frame_2, frame_2, frame_2])


for i in range(srcFeatures.shape[0]):
     cv2.circle(out, (int(srcFeatures[i][0]),int(srcFeatures[i][1])), 4, (0, 255, 255), 1)   
     cv2.circle(out, (int(dstFeatures[i][0])+cols1,int(dstFeatures[i][1])), 4, (0, 255, 255), 1)
     cv2.line(out, (int(srcFeatures[i][0]),int(srcFeatures[i][1])), (int(dstFeatures[i][0])+cols1,int(dstFeatures[i][1])), (0, 255, 0), 1)
     
for i in range(inlierSrc.shape[0]):
     cv2.circle(out, (int(inlierSrc[i][0]),int(inlierSrc[i][1])), 4, (255, 0, 255), 1)   
     cv2.circle(out, (int(inlierDst[i][0])+cols1,int(inlierDst[i][1])), 4, (255, 0, 255), 1)
     cv2.line(out, (int(inlierSrc[i][0]),int(inlierSrc[i][1])), (int(inlierDst[i][0])+cols1,int(inlierDst[i][1])), (255, 0, 0), 1)
     
plt.imshow(out)
plt.show()
