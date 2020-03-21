#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:31:03 2020

@author: zhengsc
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.,2.,3.,4.,5.])
y = np.array([1.,3.,2.,3.,5,])

plt.scatter(x,y)
plt.axis([0,6,0,6])
plt.show()

# 首先要计算x和y的均值
x_mean = np.mean(x)
y_mean = np.mean(y)

# 最小二乘法求参数
# a的分子num、分母d
num = 0.0
d = 0.0
for x_i,y_i in zip(x,y):   # zip函数打包成[(x_i,y_i)...]的形式
    num = num + (x_i - x_mean) * (y_i - y_mean)
    d = d + (x_i - x_mean) ** 2
a = num / d
b = y_mean - a * x_mean

# 拟合函数
y_hat = a * x + b
plt.scatter(x,y)    # 绘制散点图
plt.plot(x,y_hat,color='r')    # 绘制直线
plt.axis([0,6,0,6])
plt.show()

# 预测
x_predict = 6
y_predict = a * x_predict + b
print(y_predict)