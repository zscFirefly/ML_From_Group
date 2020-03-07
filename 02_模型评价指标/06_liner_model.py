#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:00:24 2020

@author: reocar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error # sklearn中计算均方根误差的包
from sklearn.metrics import mean_absolute_error # sklearn中计算平均绝对误差的包
from sklearn.metrics import r2_score # sklearn中计算R方的包

# 查看数据集描述
boston = datasets.load_boston()
print("数据集特征",boston.DESCR)
print("数据集特征列：\n",boston.feature_names)

# 取出数据中的第六例的所有行（房间数量）
x = boston.data[:,5]
y = boston.target

plt.scatter(x,y)
plt.show()

# 数据清洗
np.max(y)
# 这里有一个骚操作，用比较运算符返回一个布尔值的向量，将其作为索引，直接在矩阵里对每个元素进行过滤。
x = x[y < 50.0]
y = y[y < 50.0]
plt.scatter(x,y)
plt.show()

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# 线性回归
reg = LinearRegression()
reg.fit(x_train,y_train)
# 输出线性回归参数
print(reg.coef_)
plt.scatter(x_train,y_train)
plt.plot(x_train, reg.predict(x_train),color='r')
plt.show()

# 测试集预测
y_predict = reg.predict(x_test)
# 计算 MSE均方误差
mse_test = np.sum((y_predict - y_test) ** 2) / len(y_test)
print("MSE均方误差",mse_test)
# 计算 RMSE均方根误差
rmse_test = sqrt(mse_test)
print("MSE均方根误差",rmse_test)
# 计算 MAE平均绝对误差
mae_test = np.sum(np.absolute(y_predict - y_test)) / len(y_test)
print("MAE平均绝对误差",mae_test)


# 调用sklearn的包
print("sklearn调包结果.....")
mse_test = mean_squared_error(y_test, y_predict)
print("MSE均方误差",mse_test)
mae_test = mean_absolute_error(y_test, y_predict)
print("MAE平均绝对误差",mae_test)
r2 = r2_score(y_test, y_predict)
print("R方",r2)