# -*- coding: utf-8 -*-
"""
Created on Sun May  3 01:14:20 2020

@author: Praveen
"""

import numpy as np

def essentialMatrix(F, K):
    E = np.matmul(K.T, np.dot(F, K))
    u, s, v = np.linalg.svd(E, full_matrices = True)
    s_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype = np.float32)
    E = np.matmul(u, np.matmul(s_new, v))
    
    return E