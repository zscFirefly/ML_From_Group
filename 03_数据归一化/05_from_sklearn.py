#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:47:30 2020

@author: zhengsc
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=666)

# 数据归一化
standardScaler = StandardScaler()
# 归一化的过程跟训练模型一样
standardScaler.fit(X_train)
standardScaler.mean_
standardScaler.scale_   # 表述数据分布范围的变量，替代std_

# 使用transform进行归一化
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)