#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:49:50 2020

@author: zhengsc
"""

import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None    # 系数（theta0~1 向量）
        self.interception_ = None   # 截距（theta0 数）
        self._theta = None  # 整体计算出的向量theta

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
        y_predict = self.predict(self, X_test)
        return r2_score(y_test, y_predict)
    

    def __repr__(self):
        return "LinearRegression()"