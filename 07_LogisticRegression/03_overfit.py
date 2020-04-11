#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 02:00:40 2020

@author: zhengsc
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def plot_decision_boundary(model, axis):  
    '''可视化决策平面'''
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

def PolynomialLogisticRegression02(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C))
    ])

def PolynomialLogisticRegression03(degree, C, penalty):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C, penalty=penalty))
    ])


if __name__ == "__main__":
    np.random.seed(666)
    # 构建服从标准差为0，方差为1的分布，200个样本，有两个特征
    X = np.random.normal(0, 1, size=(200, 2))
    # 构建输入空间X与标签y的关系：是一个抛物线，通过布尔向量转为int类型
    y = np.array((X[:,0]**2+X[:,1])<1.5, dtype='int')
    # 随机在样本中挑20个点，强制分类为1（相当于噪音）
    for _ in range(20):
        y[np.random.randint(200)] = 1
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    
    print("\n正常逻辑回归")
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    print("训练集准确率:",log_reg.score(X_train, y_train))
    print("测试集准确率：",log_reg.score(X_test, y_test))
    print("逻辑回归可视化决策平面")
    plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.show()
    
    print("\n多项式逻辑回归")
    poly_log_reg = PolynomialLogisticRegression(degree=2)
    poly_log_reg.fit(X_train, y_train)
    print("训练集准确率:",poly_log_reg.score(X_train, y_train))
    print("测试集准确率：",poly_log_reg.score(X_test, y_test))
    print("多项式逻辑回归可视化决策平面")
    plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.show()
    
    # 修改degree参数，使其过拟合
    print("\n过拟合+多项式逻辑回归")
    poly_log_reg2 = PolynomialLogisticRegression(degree=20)
    poly_log_reg2.fit(X_train, y_train)
    poly_log_reg2.score(X_train, y_train)
    print("训练集结果",poly_log_reg2.score(X_train, y_train))
    print("测试集准确率：",poly_log_reg2.score(X_test, y_test))
    print("过拟合+多项式逻辑回归可视化决策平面")
    plot_decision_boundary(poly_log_reg2, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.show()
    
    print("\n过拟合+超参数+多项式逻辑回归")
    poly_log_reg3 = PolynomialLogisticRegression02(degree=20, C=0.1)
    poly_log_reg3.fit(X_train, y_train)
    print("训练集结果",poly_log_reg3.score(X_train, y_train))
    print("测试集准确率：",poly_log_reg3.score(X_test, y_test))
    print("过拟合+超参数+多项式逻辑回归可视化决策平面")
    plot_decision_boundary(poly_log_reg3, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.show()
    
    print("\n过拟合+正则+超惨+多项式逻辑回归")
    poly_log_reg4 = PolynomialLogisticRegression03(degree=20, C=0.1, penalty='l2')
    poly_log_reg4.fit(X_train, y_train)
    poly_log_reg4.score(X_train, y_train)  
    print("过拟合+正则+超惨+多项式逻辑回归可视化决策平面")
    plot_decision_boundary(poly_log_reg4, axis=[-4, 4, -4, 4])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.show()