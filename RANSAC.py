# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:15:23 2020

@author: Praveen
"""

import numpy as np
from EstimateFundamentalMatrix import fundamentalMatrix

def outlierRejection(srcFeatures, dstFeatures):
    Best_count = 0
    M = 250
    for i in range(M):
        sampled_index = np.random.randint(0, srcFeatures.shape[0], size=8)
        F = fundamentalMatrix(srcFeatures[sampled_index, :],
                                      dstFeatures[sampled_index, :])
        src = []
        dst = []
        S = 0
        for j in range(srcFeatures.shape[0]):
            matchesSrc = np.append(srcFeatures[j, :], 1)
            matchesDst = np.append(dstFeatures[j, :], 1)
            if abs(np.matmul(np.matmul(matchesDst.T, F), matchesSrc)) < 0.01:
                src.append(srcFeatures[j, :])
                dst.append(dstFeatures[j, :])
                S = S + 1

        if S > Best_count:
            Best_count = S
            F_best = F
            inliersSrc = src
            inliersDst = dst

    inliersSrc = np.array(inliersSrc)
    inliersDst = np.array(inliersDst)

    return F_best, inliersSrc, inliersDst

