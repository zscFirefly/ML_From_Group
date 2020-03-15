#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:23:53 2020

@author: reocar
"""

import numpy as np
import matplotlib.pyplot as plt

# Max-Min归一化
# 最值归一化（向量）
x = np.random.randint(0,100,size=100)
x = (x - np.min(x)) / (np.max(x) -  np.min(x))

# 最值归一化（矩阵）
X = np.random.randint(0,100,(50,2)) # 0～100范围内的50*2的矩阵
X = np.array(X, dtype=float) # 将矩阵改为浮点型
X = np.random.randint(0,100,(50,2))
X_MM=np.zeros((50,2)) # 创建一个同样大小的矩阵
X_MM[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0])) # X[:,0]第一列，第一个特征
X_MM[:,1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1])) # X[:,1]第二列，第二个特征

# 如果有n个特征，可以写个循环：
for i in range(0,2):
    X[:,i] = (X[:,i]-np.min(X[:,i])) / (np.max(X[:,i] - np.min(X[:,i])))
    
plt.scatter(X[:,0],X[:,1],c='r')
plt.show()
plt.scatter(X_MM[:,0],X_MM[:,1],c='g')
plt.show()



# 均值方差归一化（把数据集的标准差转化为1）
X2 = np.array(np.random.randint(0,100,(50,2)),dtype=float)
X_S = np.zeros((50,2))

# 套用公式，对每一列做均值方差归一化
for i in range(0,2):
    X_S[:,i]=(X2[:,i]-np.mean(X2[:,i])) / np.std(X2[:,i])
    
plt.scatter(X2[:,0],X2[:,1],c='r')
plt.show()
plt.scatter(X_S[:,0],X_S[:,1],c='g')
plt.show()

