# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:15:23 2020

@author: Praveen
"""

import numpy as np
from EstimateFundamentalMatrix import fundamentalMatrix

def outlierRejection(srcFeatures, dstFeatures):
    M = 10000;
    index = np.random.randint(srcFeatures.shape[0], size = (M, 8))
    epsilon = 1e-3
    n = np.zeros(M);
    src = np.ones((3, srcFeatures.shape[0]))
    dst = np.ones((3, srcFeatures.shape[0]))
    src[0:2, :] = srcFeatures.T
    dst[0:2, :] = dstFeatures.T
    
    for i in range(M):
        F = fundamentalMatrix(srcFeatures[index[i, :], :], dstFeatures[index[i, :], :])
        S = np.zeros(srcFeatures.shape[0])
        for j in range(srcFeatures.shape[0]):
            S[j] = np.dot(np.dot(dst[:, j].T, F), src[:, j])
        inlierCorrespondance = np.absolute(S) < epsilon
        n[i] = np.sum(inlierCorrespondance + np.zeros(srcFeatures.shape[0]), axis = None)
    
    n_best = np.argsort(-n)[0]
    F_best = fundamentalMatrix(srcFeatures[index[n_best, :], :], dstFeatures[index[n_best, :], :])
    S = np.zeros(srcFeatures.shape[0])
    for j in range(srcFeatures.shape[0]):
    	S[j] = np.dot(np.dot(dst[:, j].T, F_best), src[:, j])
    inlierThreshold = np.absolute(S)
    index_best = np.argsort( inlierThreshold)
    inlier_srcFeatures = srcFeatures[index_best]
    inlier_dstFeatures = dstFeatures[index_best]
    return F_best, inlier_srcFeatures, inlier_dstFeatures
    
    
    