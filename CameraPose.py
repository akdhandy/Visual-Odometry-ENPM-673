# -*- coding: utf-8 -*-
"""
Created on Sun May  3 14:29:59 2020

@author: Praveen
"""

import numpy as np

def obtainCameraPose(E):
    u, s, v = np.linalg.svd(E, full_matrices = True)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype = np.float32)
    cameraCenter = []
    rotationMatrix = []
    
    cameraCenter.append(u[:, 2])
    cameraCenter.append(-u[:, 2])
    cameraCenter.append(u[:, 2])
    cameraCenter.append(-u[:, 2])
    
    rotationMatrix.append(np.matmul(u, np.dot(W, v)))
    rotationMatrix.append(np.matmul(u, np.dot(W, v)))
    rotationMatrix.append(np.matmul(u, np.dot(W.T, v)))
    rotationMatrix.append(np.matmul(u, np.dot(W.T, v)))
    
    for i in range(4):
        if np.linalg.det(rotationMatrix[i]) < 0:
            cameraCenter[i] = - cameraCenter[i]
            rotationMatrix[i] = - rotationMatrix[i]
            
    return cameraCenter, rotationMatrix