'''
Author: jirufengyu
Date: 2020-11-02 07:53:07
LastEditTime: 2020-11-02 08:21:28
LastEditors: jirufengyu
Description: Nothing
FilePath: /Signal-1/DSNMF/test.py
'''
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.cluster import KMeans
from dsnmf import DSNMF, appr_seminmf
from scipy.io import loadmat
mat = loadmat('/home/stu2/Signal-1/DSNMF/PIE_pose27.mat', struct_as_record=False, squeeze_me=True)

data, gnd = mat['fea'].astype('float32'), mat['gnd']

# Normalise each feature to have an l2-norm equal to one.
data /= np.linalg.norm(data, 2, 1)[:, None]
n_classes = np.unique(gnd).shape[0]
kmeans = KMeans(n_classes, precompute_distances=False)
dsnmf = DSNMF(data, layers=(400, 100))
dsnmf.train(data, layers=(400, 100))
fea=dsnmf.H
pred = kmeans.fit_predict(fea)
score = sklearn.metrics.normalized_mutual_info_score(gnd, pred)

print("NMI: {:.2f}%".format(100 * score))