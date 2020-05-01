#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:02:19 2020

@author: zhengsc
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

boston = datasets.load_boston()
X = boston.data
y = boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)



def StandardLinearSVR(epsilon = 0.1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('linearSVR', LinearSVR(epsilon=epsilon))
    ])
    
svr = StandardLinearSVR()
svr.fit(X_train,y_train)
print("svm准确率：",svr.score(X_test, y_test))