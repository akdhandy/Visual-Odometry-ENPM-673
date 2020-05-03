# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:03:56 2020

@author: Praveen
"""

import numpy as np

def fundamentalMatrix(srcFeatures, dstFeatures):
    A = np.ones((8, 9))
    Ta = np.zeros((3,3))
    Tb = np.zeros((3,3))
    
    srcFeatures_centroid = np.average(srcFeatures, axis = 0)
    dstFeatures_centroid = np.average(dstFeatures, axis = 0)
    mean_subtracted_srcFeatures_centroid = srcFeatures - srcFeatures_centroid.reshape(1,2)
    mean_subtracted_dstFeatures_centroid = dstFeatures - dstFeatures_centroid.reshape(1,2)
    srcScale = (2/(np.sum(mean_subtracted_srcFeatures_centroid**2, axis = None)/8))**0.5
    dstScale = (2/(np.sum(mean_subtracted_dstFeatures_centroid**2, axis = None)/8))**0.5
    
    Ta[0, 0] = srcScale
    Ta[0, 2] = -srcScale*srcFeatures_centroid[0]
    Ta[1, 1] = srcScale
    Ta[1, 2] = -srcScale*srcFeatures_centroid[1]
    Ta[2, 2] = 1
    
    Tb[0, 0] = dstScale
    Tb[0, 2] = -dstScale*dstFeatures_centroid[0]
    Tb[1, 1] = dstScale
    Tb[1, 2] = -dstScale*dstFeatures_centroid[1]
    Tb[2, 2] = 1
    
    
    normalized_srcFeatures = mean_subtracted_srcFeatures_centroid*srcScale
    normalized_dstFeatures = mean_subtracted_dstFeatures_centroid*dstScale
    
    A[:, 0:2] = normalized_srcFeatures*normalized_dstFeatures[:, 0].reshape(8, 1)
    A[:, 2] = normalized_dstFeatures[:, 0]
    A[:, 3:5] = normalized_srcFeatures*normalized_dstFeatures[:, 1].reshape(8, 1)
    A[:, 5] = normalized_dstFeatures[:, 1]
    A[:, 6:8] = normalized_srcFeatures
    
    u, s, v = np.linalg.svd(A, full_matrices = True)
    F_normalized = v[8,:].reshape(3,3)
    uf, sf, vf = np.linalg.svd(F_normalized, full_matrices = True)
    sf[2] = 0
    sf_new = np.diag(sf)
    F_normalized = np.dot(uf, np.dot(sf_new, vf))
    F = np.dot(Tb.T, np.dot(F_normalized, Ta))
    return F
    
    
    
    