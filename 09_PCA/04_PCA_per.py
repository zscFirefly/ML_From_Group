#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:58:37 2020

@author: zhengsc
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

print("降纬前处理结果")
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print("测试集准确率",knn_clf.score(X_test, y_test))

print("降纬后处理结果")
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train) # 训练数据集降维结果
X_test_reduction = pca.transform(X_test) # 测试数据集降维结果

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print("测试集准确率",knn_clf.score(X_test_reduction, y_test))

print("主成分解释方差比例",pca.explained_variance_ratio_)



print("绘制不同主成分数量解释方差的比例")

pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
print("不同主成分解释方差的比例",pca.explained_variance_ratio_)
plt.plot([i for i in range(X_train.shape[1])], 
         [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()

print("利用百分比求主成分")
pca = PCA(0.95)
res = pca.fit(X_train)
print(res)
print("主成分个数",pca.n_components_)


X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print("主成分处理后测试集准确率",knn_clf.score(X_test_reduction, y_test))

print("降维可视化")
pca = PCA(n_components=2)
pca.fit(X)
X_reduction = pca.transform(X)
for i in range(10):
    plt.scatter(X_reduction[y==i,0], X_reduction[y==i,1], alpha=0.8)
plt.show()