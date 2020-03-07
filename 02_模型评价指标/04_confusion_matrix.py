#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:10:37 2020

@author: zhengsc
"""


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    # (y_true == 0)：向量与数值按位比较，得到的是一个布尔向量
    # 向量与向量按位与，结果还是布尔向量
    # np.sum 计算布尔向量中True的个数(True记为1，False记为0)
    return np.sum((y_true == 0) & (y_predict == 0))  # 向量与向量按位与，结果还是向量

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    # (y_true == 0)：向量与数值按位比较，得到的是一个布尔向量
    # 向量与向量按位与，结果还是布尔向量
    # np.sum 计算布尔向量中True的个数(True记为1，False记为0)
    return np.sum((y_true == 0) & (y_predict == 1))  # 向量与向量按位与，结果还是向量

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    # (y_true == 0)：向量与数值按位比较，得到的是一个布尔向量
    # 向量与向量按位与，结果还是布尔向量
    # np.sum 计算布尔向量中True的个数(True记为1，False记为0)
    return np.sum((y_true == 1) & (y_predict == 0))

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    # (y_true == 0)：向量与数值按位比较，得到的是一个布尔向量
    # 向量与向量按位与，结果还是布尔向量
    # np.sum 计算布尔向量中True的个数(True记为1，False记为0)
    return np.sum((y_true == 1) & (y_predict == 1))  # 向量与向量按位与，结果还是向量


def confusion_matrix(TN,FP,FN,TP):
    return np.array([
        [TN, FP],
        [FN, TP]
    ])
 
def recall_score(y_true, y_predict):
    tp = np.sum((y_true == 1) & (y_predict == 1))
    fn = np.sum((y_true == 1) & (y_predict == 0))
    try:
        return tp / (tp + fn)
    except:
        return 0.0     

def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0  

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
    print(y_log_predict)
    # 计算召回率
    recall_score = recall_score(y_test, y_log_predict)
    print(recall_score)

    # 方法二：对逻辑回归设置阀值
    # decision_scores = log_reg.decision_function(X_test)
    # print(decision_scores)
    # y_log_predict = np.array(decision_scores >= 5, dtype='int') # 对逻辑回归设置阀值
    # recall_score(y_test, y_log_predict)

    acc = log_reg.score(X_test, y_test) #逻辑回归准确度
    
    print("准确率:",acc)
    
    # 计算混淆矩阵    
    TN = TN(y_test, y_log_predict)
    
    FP = FP(y_test, y_log_predict)
    
    FN = FN(y_test, y_log_predict)
    
    TP = TP(y_test, y_log_predict)
    
    confusion_matrix = confusion_matrix(TN,FP,FN,TP)

    print(TN,FP,FN,TP)
    print("混淆矩阵")
    print(confusion_matrix)
    print("召回率")
    print(recall_score)