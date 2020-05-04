# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:03:56 2020

@author: Praveen
"""

import numpy as np

def fundamentalMatrix(srcFeatures, dstFeatures):
    points = srcFeatures.shape[0]
    A = []
    B = np.ones((points, 1))

    srcFeatures_centroid_x = np.sum(srcFeatures[:, 0]) / points
    srcFeatures_centroid_y = np.sum(srcFeatures[:, 1]) / points

    s = points / np.sum(((srcFeatures[:, 0] - srcFeatures_centroid_x)**2 + (srcFeatures[:, 1] - srcFeatures_centroid_y)**2)**(1 / 2))
    T_a = np.matmul(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -srcFeatures_centroid_x], [0, 1, -srcFeatures_centroid_y], [0, 0, 1]]))

    srcFeatures = np.array(srcFeatures.T)
    srcFeatures = np.append(srcFeatures, B)

    srcFeatures = np.reshape(srcFeatures, (3, points))
    srcFeatures = np.matmul(T_a, srcFeatures)
    srcFeatures = srcFeatures.T

    dstFeatures_centroid_x = np.sum(dstFeatures[:, 0]) / points
    dstFeatures_centroid_y = np.sum(dstFeatures[:, 1]) / points

    s = points / np.sum(((dstFeatures[:, 0] - dstFeatures_centroid_x)**2 + (dstFeatures[:, 1] - dstFeatures_centroid_y)**2)**(1 / 2))
    T_b = np.matmul(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -dstFeatures_centroid_x], [0, 1, -dstFeatures_centroid_y], [0, 0, 1]]))

    dstFeatures = np.array(dstFeatures.T)
    dstFeatures = np.append(dstFeatures, B)

    dstFeatures = np.reshape(dstFeatures, (3, points))
    dstFeatures = np.matmul(T_b, dstFeatures)
    dstFeatures = dstFeatures.T

    for i in range(points):
        u_a = srcFeatures[i, 0]
        v_a = srcFeatures[i, 1]
        u_b = dstFeatures[i, 0]
        v_b = dstFeatures[i, 1]
        A.append([u_a * u_b, v_a * u_b, u_b, u_a * v_b, v_a * v_b, v_b, u_a, v_a, 1])

    _, _, v = np.linalg.svd(A)
    F = v[-1]

    F = np.reshape(F, (3, 3)).T
    F = np.matmul(T_a.T, F)
    F = np.matmul(F, T_b)

    F = F.T
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = np.matmul(U, S)
    F = np.matmul(F, V)
    if F[2, 2] > 1e-6:
        F = F / F[2, 2]

    return F
