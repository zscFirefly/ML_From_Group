#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:42:33 2020

@author: zhengsc
"""


import numpy as np
import matplotlib.pyplot as plt

X = np.empty([100,2])
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0., 10., size=100)

plt.scatter(X[:,0],X[:,1])
plt.show()


def demean(X):
    # axis=0按列计算均值，即每个属性的均值，1则是计算行的均值
    return (X - np.mean(X, axis=0))

def f(w,X):
    # 定义目标函数
    return np.sum((X.dot(w)**2))/len(X)

def df_math(w,X):
    # 求梯度
    return X.T.dot(X.dot(w))*2./len(X)

# 验证梯度求解是否正确，使用梯度调试方法：
def df_debug(w, X, epsilon=0.0001):
    # 先创建一个与参数组等长的向量
    res = np.empty(len(w))
    # 对于每个梯度，求值
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res

def direction(w):
    return w / np.linalg.norm(w)


def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    # 梯度上升法代码
    w = direction(initial_w)
    cur_iter = 0
    while cur_iter < n_iters:
        gradient = df_math(w,X)
        last_w = w
        w = last_w + eta * gradient
        w = direction(w)    # 将w转换成单位向量
        if (abs(f(w,X) - f(last_w,X)) < epsilon):
            break
        cur_iter += 1
    return w

X_demean = demean(X)
# 注意看数据分布没变，但是坐标已经以原点为中心了
plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.show()


initial_w = np.random.random(X.shape[1])
eta = 0.001

w = gradient_ascent(df_debug, X_demean, initial_w, eta)
plt.scatter(X_demean[:,0],X_demean[:,1])
plt.plot([0,w[0]*30],[0,w[1]*30], color='red')
plt.show()



def first_n_component(n, X, eta=0.001, n_iters=1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)    
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = gradient_ascent(df_math, X_pca, initial_w, eta)
        res.append(w)
        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w   
    return res

res = first_n_component(2,X)
print(res)