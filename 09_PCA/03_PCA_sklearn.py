#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:53:40 2020

@author: a111
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 10., size=100)


plt.scatter(X[:,0],X[:,1])


# 初始化实例对象，传入主成分个数
pca = PCA(n_components=2)
pca.fit(X)
# 求解特征向量
w0 = pca.components_[0]
w1 = pca.components_[1]
print("第一主成分向量：",w0)
print("第二主成分向量：",w1)
plt.plot([0,w0[0]*30],[0,w0[1]*30], color='red')
plt.plot([0,w1[0]*30],[0,w1[1]*30], color='green')
plt.show()

# 数据集降维
X_reduction = pca.transform(X)
print("降维后数据纬度",X_reduction.shape)