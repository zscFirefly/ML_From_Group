#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:40:55 2020

@author: zhengsc
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from Metrics import accuracy_score # 调用自己封装的方法
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
    

if __name__ == "__main__":
    digits = datasets.load_digits() # 加载手写数字数据集
    digits.keys() # 数据集的存储内容，以json存储
#    print(digits.DESCR)   
    X = digits.data 
    y = digits.target

    # 简单查看数据集的demo
#    digits.target_names
#    some_digit = X[666] # 随机取一个样本
#    some_digmit_image = some_digit.reshape(8, 8)
#    plt.imshow(some_digmit_image, cmap = matplotlib.cm.binary)
#    plt.show()
#    y[666]
    
    # knn进行分类
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=666)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    y_predict = knn_clf.predict(X_test)
    acc = accuracy_score(y_test,y_predict)
    print(acc)
    # 直接调用检验预测的方法        
    acc = knn_clf.score(X_test,y_test)
    
    
    # 指定最佳值的分数，初始化为0.0；设置最佳值k，初始值为-1
    best_score = 0.0
    best_k = -1
    for k in range(1, 11):  # 暂且设定到1～11的范围内
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        print(score)
        if score > best_score:
            best_k = k
            best_score = score
    print("best_k = ", best_k)
    print("best_score = ", best_score)
    
    # 两次循环寻找参数
    best_method = ""
    best_score = 0.0
    best_k = -1
    for method in ["uniform","distance"]:
        for k in range(1, 11):
            knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method, p=2)
            knn_clf.fit(X_train, y_train)
            score = knn_clf.score(X_test, y_test)
            if score > best_score:
                best_k = k
                best_score = score
                best_method = method
    print("best_method = ", method)
    print("best_k = ", best_k)
    print("best_score = ", best_score)
    
    # 使用网格搜索
    param_search = [
        {
            "weights":["uniform"],
            "n_neighbors":[i for i in range(1,11)]
        },
        {
            "weights":["distance"],
            "n_neighbors":[i for i in range(1,11)],
            "p":[i for i in range(1,6)]
        }
    ]
    
    knn_clf = KNeighborsClassifier()# 调用网格搜索方法
    grid_search = GridSearchCV(knn_clf, param_search) # 定义网格搜索的对象grid_search，其构造函数的第一个参数表示对哪一个分类器进行算法搜索，第二个参数表示网格搜索相应的参数
    grid_search.fit(X_train, y_train)
    # 返回最优参数
    grid_search.best_estimator_

