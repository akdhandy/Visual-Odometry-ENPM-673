# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:43:41 2020

@author: Praveen
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def featureDetectionMatching(frame_1, frame_2):
    orbDetector = cv2.ORB_create()
    keypoints_1, descriptor_1 = orbDetector.detectAndCompute(frame_1, None)
    keypoints_2, descriptor_2 = orbDetector.detectAndCompute(frame_2, None)
    
    index_params = dict(algorithm = 6,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 2)
    
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    featureMatching = flann.knnMatch(descriptor_1, descriptor_2, k = 2)
    
    srcFeatures = []
    dstFeatures = []
    matches = [[0,0] for i in range(len(featureMatching))]
    
    for frame, (m, n) in enumerate(featureMatching):
        if m.distance < 0.75*n.distance:
            matches[frame] = [1, 0]
            srcFeatures.append(keypoints_1[m.queryIdx].pt)
            dstFeatures.append(keypoints_2[m.trainIdx].pt)
    
    srcFeatures = np.array(srcFeatures).reshape(-1, 2)
    dstFeatures = np.array(dstFeatures).reshape(-1, 2)
    '''
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matches,
                   flags = 0)

    img3 = cv2.drawMatchesKnn(frame_1, keypoints_1, frame_2, keypoints_2, featureMatching, None, **draw_params)

    plt.imshow(img3)
    plt.show()
    '''
    return srcFeatures, dstFeatures




            
            
    
            
            
    
    
    
