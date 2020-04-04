#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:45:19 2020

@author: zhengsc
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# 多元回归
print("绘制多元回归")
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 + x**2 + x + 2 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)

lin_reg = LinearRegression()
def PolynomialRegression(degree):
    return Pipeline([
        ('poly',PolynomialFeatures(degree)),
        ('std_scaler',StandardScaler()),
        ('lin_reg',lin_reg)
    ])
np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X,y)

poly30_reg = PolynomialRegression(degree=30)
poly30_reg.fit(X_train,y_train)
y30_predict = poly30_reg.predict(X_test)
mean_squared_error(y_test,y30_predict)
# 输出：2.8492121876294063
X_plot = np.linspace(-3,3,100).reshape(100,1)
y_plot = poly30_reg.predict(X_plot)
plt.scatter(X,y)
plt.plot(X_plot[:,0],y_plot,color='r')
plt.axis([-3,3,0,10])
plt.show()


print("绘制Lasso回归")
# Lasso回归
def LassoRegression(degree,alpha):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('lasso_reg',Lasso(alpha=alpha))
    ])


lasso_reg1 = LassoRegression(30,0.1) # 第二个参数为权重系数
lasso_reg1.fit(X_train,y_train)
y1_predict=lasso_reg1.predict(X_test)
mean_squared_error(y_test,y1_predict)
# 输出：0.8965099322966458

X_plot = np.linspace(-3,3,100).reshape(100,1)
y_plot = lasso_reg1.predict(X_plot)
plt.scatter(X,y)
plt.plot(X_plot[:,0],y_plot,color='r')
plt.axis([-3,3,0,10])
plt.show()


print("绘制岭回归")
# 岭回归
# 需要传入一个多项式项数的参数degree以及一个alpha值
def ridgeregression(degree,alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("standard", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))   #alpha值就是正则化那一项的系数
    ])

ridge1_reg = ridgeregression(degree=30,alpha=1) #调整alpha系数，控制权重
ridge1_reg.fit(X_train,y_train)
y1_predict = ridge1_reg.predict(X_test)
mean_squared_error(y_test,y1_predict)
# 输出：1.00677921937119

X_plot = np.linspace(-3,3,100).reshape(100,1)
y_plot = ridge1_reg.predict(X_plot)
plt.scatter(X,y)
plt.plot(X_plot[:,0],y_plot,color='g')
plt.axis([-3,3,0,10])
plt.show()