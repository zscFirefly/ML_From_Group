#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:02:40 2020

@author: reocar
"""

# 等距分厢
# 等频分箱
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
df = pd.DataFrame([[22,1],[13,1],[33,1],[52,0],[16,0],[42,1],[53,1],[39,1],[26,0],[66,0]],columns=['age','Y'])
df['age_bin_2'] = pd.cut(df['age'],3)  #等距分箱
df['age_bin_1'] = pd.qcut(df['age'],3) #等频分箱
display(df)


# k-mean分箱（待修改）
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=666)
kmodel=KMeans(n_clusters=2)  #k为聚成几类
kmodel.fit(X_train[:,0].reshape(len(X_train[:,0]),1)) #训练模型
c=pd.DataFrame(kmodel.cluster_centers_) #求聚类中心
c=c.sort_values(by=0) #排序
w=c.rolling(2).mean().iloc[1:]#用滑动窗口求均值的方法求相邻两项求中点，作为边界点
#w=[0] +list(w[0] + X_train.max())  #把首末边界点加上
w = [0,w[0],X_train.max()]
d3= pd.cut(X_train,w,labels=range(2))

# 二值化
from sklearn.preprocessing import Binarizer
# Binarizer函数也可以设定一个阈值，结果数据值大于阈值的为1，小于阈值的为0
binarizer = Binarizer(threshold=0.0).fit(X_train)
binarizer.transform(X_train)