#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:49:50 2020

@author: zhengsc
"""

import numpy as np
# from .metrics import r2_score
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None    # 系数（theta0~1 向量）
        self.interception_ = None   # 截距（theta0 数）
        self._theta = None  # 整体计算出的向量theta


    def mean_squared_error(self,y_true, y_predict):
        """计算y_true和y_predict之间的MSE"""
        assert len(y_true) == len(y_predict), \
            "the size of y_true must be equal to the size of y_predict"
        return np.sum((y_true - y_predict) ** 2) / len(y_true)

    def r2_score(self,y_true, y_predict):
        """计算y_true和y_predict之间的准确率"""
        assert y_true.shape[0] == y_predict.shape[0],  "the size of y_true must be equal to the size of y_predict"
        return 1 - self.mean_squared_error(y_test, y_predict) / np.var(y_test)

    def fit_normal(self, X_train, y_train):
        """根据训练数据X_train，y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # 正规化方程求解
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
            
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    def predict(self, X_predict):
        """给定待预测的数据集X_predict，返回表示X_predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        y_predict = X_b.dot(self._theta)
        return y_predict

    def score(self, X_test, y_test):
        """很倔测试机X_test和y_test确定当前模型的准确率"""
        y_predict = self.predict(X_test)
        return self.r2_score(y_test, y_predict)
    

    def __repr__(self):
        return "LinearRegression()"

if __name__ == '__main__':
    linreg = LinearRegression()
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y<50.0]
    y = y[y<50.0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=532)#选择20%为测试集

    model = linreg.fit_normal(X_train, y_train)
    print('模型参数:')
    print(model)
    # 训练后模型截距
    print('模型截距:')
    print(linreg.interception_)
    # 训练后模型权重（特征个数无变化）
    print('参数权重:')
    print (linreg.coef_)
    # y_pred = linreg.predict(X_test)
    print(linreg.score(X_test,y_test))
