#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:56:45 2020

@author: zhengsc
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split #这里是引用了交叉验证
from sklearn.linear_model import LinearRegression  #线性回归

# 数据集准备
boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y<50.0]
y = y[y<50.0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=532)#选择20%为测试集
# 线性回归
linreg = LinearRegression()
#训练
model = linreg.fit(X_train, y_train)
print('模型参数:')
print(model)
# 训练后模型截距
print('模型截距:')
print(linreg.intercept_)
# 训练后模型权重（特征个数无变化）
print('参数权重:')
print (linreg.coef_)
y_pred = linreg.predict(X_test)

sum_mean = 0
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - y_test[i]) ** 2
sum_erro = np.sqrt(sum_mean /len(y_pred))  # 测试级的数量
# calculate RMSE
print ("RMSE by hand:", sum_erro)

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
plt.plot(range(len(y_pred)), y_test, 'r', label="test")
plt.legend(loc="upper right")  # 显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()
 
