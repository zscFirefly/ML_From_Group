#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 01:30:24 2020

@author: zhengsc
"""


import numpy as np
# 因为逻辑回归是分类问题，因此需要对评价指标进行更改
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LogisticRegression:

    def __init__(self):
        """初始化Logistic Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None
        
    def accuracy_score(self,y_true, y_predict):
        """计算y_true和y_predict之间的准确率"""
        assert y_true.shape[0] == y_predict.shape[0],  "the size of y_true must be equal to the size of y_predict"
        return sum(y_true == y_predict) / len(y_true)

    """
    定义sigmoid方法
    参数：线性模型t
    输出：sigmoid表达式
    """
    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))
    
    """
    fit方法，内部使用梯度下降法训练Logistic Regression模型
    参数：训练数据集X_train, y_train, 学习率, 迭代次数
    输出：训练好的模型
    """
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        """
        定义逻辑回归的损失函数
        参数：参数theta、构造好的矩阵X_b、标签y
        输出：损失函数表达式
        """
        def J(theta, X_b, y):
            # 定义逻辑回归的模型：y_hat
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                # 返回损失函数的表达式
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')
        """
        损失函数的导数计算
        参数：参数theta、构造好的矩阵X_b、标签y
        输出：计算的表达式
        """
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

        """
        梯度下降的过程
        """
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
        # 梯度下降的结果求出参数heta
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        # 第一个参数为截距
        self.intercept_ = self._theta[0]
        # 其他参数为各特征的系数
        self.coef_ = self._theta[1:]
        return self

    """
    逻辑回归是根据概率进行分类的，因此先预测概率
    参数：输入空间X_predict
    输出：结果概率向量
    """
    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        # 将梯度下降得到的参数theta带入逻辑回归的表达式中
        return self._sigmoid(X_b.dot(self._theta))

    """
    使用X_predict的结果概率向量，将其转换为分类
    参数：输入空间X_predict
    输出：分类结果
    """
    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        # 得到概率
        proba = self.predict_proba(X_predict)
        # 判断概率是否大于0.5，然后将布尔表达式得到的向量，强转为int类型，即为0-1向量
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return self.accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
    
    def x2(self,x1):
        '''计算决策平面'''
        return (-self.coef_[0] * x1 - self.intercept_) / self.coef_[1]


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[y<2,:2]
    y = y[y<2]
    plt.scatter(X[y==0,0], X[y==0,1], color="red")
    plt.scatter(X[y==1,0], X[y==1,1], color="blue")
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    # 查看训练数据集分类准确度
    print("训练数据集分类准确度",log_reg.score(X_test, y_test))
    # 查看逻辑回归得到的概率
    print("逻辑回归得到的概率",log_reg.predict_proba(X_test))
    # 得到逻辑回归分类结果
    print("逻辑回归分类结果",log_reg.predict(X_test))
    
    print("绘制决策平面")
    x1_plot = np.linspace(4, 8, 1000)
    x2_plot = log_reg.x2(x1_plot)
    plt.scatter(X[y==0,0], X[y==0,1], color="red")
    plt.scatter(X[y==1,0], X[y==1,1], color="blue")
    plt.plot(x1_plot, x2_plot)