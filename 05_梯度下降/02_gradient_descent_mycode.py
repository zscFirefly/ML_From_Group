#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:35:06 2020

@author: zhengsc
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def dLF(theta):
    return derivative(lossFunction, theta, dx=1e-6)

def lossFunction(x):
    return (x-2.5)**2-1


theta = 0.0 # 初始值
eta = 0.1 # 学习率
epsilon = 1e-6 #阈值，当更新小于这个阈值当时候，break
while True:
    # 每一轮循环后，要求当前这个点的梯度是多少
    gradient = dLF(theta)
    last_theta = theta
    # 移动点，沿梯度的反方向移动步长eta
    theta = theta - eta * gradient
    # 判断theta是否达到最小值
    # 因为梯度在不断下降，因此新theta的损失函数在不断减小
    # 看差值是否达到了要求
    if(abs(lossFunction(theta) - lossFunction(last_theta)) < epsilon):
        break
print("最小值取值：",theta)
print("最小值：",lossFunction(theta))

# 绘图
# 在-1到6的范围内构建140个点
plot_x = np.linspace(-1,6,141)
# plot_y 是对应的损失函数值
plot_y = lossFunction(plot_x)
plt.plot(plot_x,plot_y)
plt.show()