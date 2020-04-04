#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:01:31 2020

@author: zhengsc
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 + x**2 + x + 2 + np.random.normal(0, 1, size=100)
print("绘制原始散点图")
plt.scatter(x, y)
plt.show()


# sklearn中对一元线性回归
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)
print("绘制一元回归结果图")
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()


# 多元线性回归
# 创建一个新的特征
(X**2).shape
# 凭借一个新的数据数据集
X2 = np.hstack([X, X**2])
# 用新的数据集进行线性回归训练
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
print("绘制多元回归结果图")
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()

print("\n新特征处理下，多元线性回归：",)
print("系数:{}".format(lin_reg2.coef_))
print("截距:{}\n".format(lin_reg2.intercept_))


# sklearn中多元线性回归
# 特征准备
# 这个degree表示我们使用多少次幂的多项式
poly = PolynomialFeatures(degree=2)    
poly.fit(X)
X2 = poly.transform(X)
X2.shape # 输出：(100, 3)
reg = LinearRegression()
reg.fit(X2, y)
y_predict = reg.predict(X2)
print("绘制sklearn多元回归结果图")
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()
print("\nsklearn中的多元线性回归：",)
print("系数:{}".format(reg.coef_))
print("截距:{}\n".format(reg.intercept_))

# 多元多项式回归
print("三元多项式回归")
X_mul = np.arange(1, 11).reshape(5, 2)
# 5行2列 10个元素的矩阵
print("X的矩阵",(X_mul.shape))
print(X_mul)
poly = PolynomialFeatures(degree=3)
poly.fit(X_mul)
# 将X转换成最多包含X二次幂的数据集
X2_mul = poly.transform(X_mul)
# 5行6列
print("转化后X的矩阵",X2_mul.shape)
print(X2_mul)



print("sklearn中的pipeline,绘图")
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('std_scale', StandardScaler()),
    ('lin_reg', LinearRegression())
])  
poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='g')
plt.show()