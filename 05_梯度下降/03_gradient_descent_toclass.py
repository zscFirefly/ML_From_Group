#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:46:49 2020

@author: zhengsc
"""
import matplotlib.pyplot as plt
from scipy.misc import derivative
import numpy as np



def lossFunction(x):
    try:
        return (x-2.5)**2-1
    except:
        return float('inf')

def dLF(theta):
    return derivative(lossFunction, theta, dx=1e-6)


def gradient_descent(initial_theta, eta, n_iters, theta_history, epsilon=1e-6):
    theta = initial_theta
    theta_history.append(theta)
    i_iters = 0
    while i_iters < n_iters:
        gradient = dLF(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if(abs(lossFunction(theta) - lossFunction(last_theta)) < epsilon):
            # 当训练小于某个阈值epsilon时候，跳出循环
            break
        i_iters += 1

def plot_theta_history(theta_history):
    plot_x = np.linspace(-1,6,141)
    # plot_y 是对应的损失函数值
    plot_y = lossFunction(plot_x)

    plt.plot(plot_x,plot_y)
    plt.plot(np.array(theta_history), lossFunction(np.array(theta_history)), color='red', marker='o')
    plt.show()

def main():
    print("开始执行")    
    eta=0.1
    n_iters = 100 # 阈值，避免死循环
    theta_history = []
    gradient_descent(0., eta, n_iters,theta_history)
    print("梯度下降完成")
    print("开始绘图")
    plot_theta_history(theta_history)
    print("梯度下降查找次数：",len(theta_history))

if __name__ == "__main__":
    main()