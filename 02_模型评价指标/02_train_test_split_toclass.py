#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:26:23 2020

@author: zhengsc
"""
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


# 封装
def train_test_split(X,y,test_ratio=0.2,seed=None):
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_train must be valid"
    
    if seed: # 是否使用随机终极，使用随机结果相同，方便debug
        np.random.seed(seed) # permutation(n) 可直接生成一个随机排列的数组，含有n个元素
    shuffle_index = np.random.permutation(len(X))
    
    test_size = int(len(X) * test_ratio)
    test_index = shuffle_index[:test_size]
    train_index = shuffle_index[test_size:]
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test=  y[test_index]
    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_ratio = 0.2)
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    
    kNN_classifier = KNeighborsClassifier(n_neighbors=3)
    kNN_classifier.fit(X_train, y_train)
    y_predict = kNN_classifier.predict(X_test)
    sum = sum(y_predict == y_test)
    accu = sum/len(y_test)
    print(accu)
    # print(sum)
    # print(len(y_test))
