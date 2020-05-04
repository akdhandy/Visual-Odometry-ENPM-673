# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:07:47 2020

@author: Praveen
"""

import numpy as np

def linearTriangulation(cameraCenter_1, cameraCenter_2, rotationMatrix_1, rotationMatrix_2, src, dst, K):
    cameraCenter_1 = cameraCenter_1.reshape((3, 1))
    cameraCenter_2 = cameraCenter_2.reshape((3, 1))
    I = np.identity(3)
    P_1 = np.matmul(K, np.matmul(rotationMatrix_1, np.hstack((I, - cameraCenter_1))))
    P_2 = np.matmul(K, np.matmul(rotationMatrix_2, np.hstack((I, - cameraCenter_2))))
    
    src = np.hstack((src, np.ones((src.shape[0], 1))))
    dst = np.hstack((dst, np.ones((dst.shape[0], 1))))
    
    X = np.zeros((src.shape[0], 3))
    
    for i in range(src.shape[0]):
        src_new = np.array([[0, -src[i, :][2], src[i, :][1]], [src[i, :][2], 0, src[i, :][0]], [src[i, :][1], src[i, :][0], 0]])
        dst_new = np.array([[0, -dst[i, :][2], dst[i, :][1]], [dst[i, :][2], 0, dst[i, :][0]], [dst[i, :][1], dst[i, :][0], 0]])
        A = np.vstack((np.matmul(src_new, P_1), np.matmul(dst_new, P_2)))
        u, s, v = np.linalg.svd(A)
        x = v[2]/v[2, 2]
        x.reshape((len(x), 1))
        X[i, :] = x[0 : 3].T
        
    return X

def obtainUniqueCameraPose(cameraCenter, rotationMatrix, X):
    ispoint3D = 0
    for i in range(4):
        count = 0
        for j in range(X[i].shape[0]):
            if (np.matmul(rotationMatrix[i][2, :], (X[i][j, :] - cameraCenter[i]))) > 0 and (X[i][j, 2] >= 0):
                count = count + 1
                
        if count > ispoint3D:
            cameraCenter_new = cameraCenter[i]
            rotationMatrix_new = rotationMatrix[i]
            X_new = X[i]
            ispoint3D = count
        
            
    return cameraCenter_new, rotationMatrix_new, X_new
                
            
        
    
    
    
    