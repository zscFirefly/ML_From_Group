#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:54:31 2020

@author: zhengsc
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC 
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y<2,:2]
y = y[y<2]

# 数据标准化
standardScaler = StandardScaler()
standardScaler.fit(X)
X_std = standardScaler.transform(X)

plt.scatter(X_std[y==0,0],X_std[y==0,1],color='red')
plt.scatter(X_std[y==1,0],X_std[y==1,1],color='blue')
plt.show()

# 线性SVM
svc = LinearSVC(C=0.1)
svc.fit(X_std,y)
print('系数：\n',svc.coef_)
print('截距',svc.intercept_)


def plot_svc_decision_boundary(model, axis):
    # 绘制svm的决策超平面
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
    # 绘制Margin区域上下两根线
    w = model.coef_[0]
    b = model.intercept_[0]
    
    # w0 * x0 + w1 * x1 + b = +1
    # w0 * x0 + w1 * x1 + b = -1
    plot_x = np.linspace(axis[0],axis[1],200)
    up_y = -w[0]/w[1] * plot_x - b/w[1] + 1/w[1]
    down_y = -w[0]/w[1] * plot_x - b/w[1] - 1/w[1]
    
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')
    
plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_std[y==0,0], X_std[y==0,1])
plt.scatter(X_std[y==1,0], X_std[y==1,1])
plt.show()