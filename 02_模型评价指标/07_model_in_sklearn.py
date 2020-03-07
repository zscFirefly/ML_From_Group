#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:36:09 2020

@author: reocar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target.copy()

    # 要构造偏斜数据，将数字9的对应索引的元素设置为1，0～8设置为0
    y[digits.target==9]=1
    y[digits.target!=9]=0
    
    # 使用逻辑回归做一个分类
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train,y_train)
    # 得到X_test所对应的预测值
    y_log_predict = log_reg.predict(X_test)
    
    recall = recall_score(y_test, y_log_predict)
    print(recall)

    # 画ROC曲线
    decision_scores = log_reg.decision_function(X_test)
    fprs, tprs, thresholds = roc_curve(y_test, decision_scores)

    # 求出AUC的值
    roc_auc_score = roc_auc_score(y_test, decision_scores)
    print(roc_auc_score)
    plt.plot(fprs, tprs)
    plt.show()