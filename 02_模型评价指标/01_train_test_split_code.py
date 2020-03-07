#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:45:35 2020

@author: zhengsc
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target

X.shape

# 数据划分 方法一
tempConcat = np.concatenate((X,y.reshape(-1,1)),axis=1) #数据合并
np.random.shuffle(tempConcat) # 打乱顺序
shuffle_X,shuffle_y = np.split(tempConcat,[4],axis=1) #数据切分
test_ratio = 0.2
test_size = int(len(X) * test_ratio) # 寻找一个划分标签
X_train = shuffle_X[test_size:]
y_train = shuffle_y[test_size:]
X_test = shuffle_X[:test_size]
y_test = shuffle_y[test_size:]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# 数据划分 方法二
shuffle_index = np.random.permutation(len(X)) # 创建一个随机的索引
test_ratio = 0.2
test_size = int(len(X) * test_ratio) # 设置一个划分比例
train_index = shuffle_index[test_size:] # 根据索引取数
test_index = shuffle_index[:test_size] # 根据索引取数
X_train = X[train_index]
X_test = X[test_index]
y_train = y[train_index]
y_test = y[test_index]

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



