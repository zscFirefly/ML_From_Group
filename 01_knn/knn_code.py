#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:22:20 2020

@author: zhengsc
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
## 数据准备
# raw_data_x是特征，raw_data_y是标签，0为良性，1为恶性
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343853454, 3.368312451],
              [3.582294121, 4.679917921],
              [2.280362211, 2.866990212],
              [7.423436752, 4.685324231],
              [5.745231231, 3.532131321],
              [9.172112222, 2.511113104],
              [7.927841231, 3.421455345],
              [7.939831414, 0.791631213]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 设置训练组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 将数据可视化
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1], color='g', label = 'Tumor Size')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1], color='r', label = 'Time')
plt.xlabel('Tumor Size')
plt.ylabel('Time')
plt.axis([0,10,0,5])
plt.show()


# 求距离
# 测试样本
x=[8.90933607318, 3.365731514]

distances = []  # 用来记录x到样本数据集中每个点的距离
for x_train in X_train:
    d = sqrt(np.sum((x_train - x) ** 2))
    distances.append(d)
# 使用列表生成器，一行就能搞定，对于X_train中的每一个元素x_train都进行前面的运算，把结果生成一个列表，与上面等价
distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
# 排序
nearest = np.argsort(distances) # 按从小到大返回索引大位置
# 选k值
k=6
topK_y = [y_train[i] for i in nearest[:k]]
# 决策规则
votes = Counter(topK_y) # 以字典的形式返回统计的频次
votes.most_common(1) # 找出票数最多的n个元素，返回一个n个元祖的列表，元组第一位对应哪个元素，第二位表示频次。


# Counter.most_common(n) 找出票数最多的n个元素，返回的是一个列表，列表中的每个元素是一个元组，元组中第一个元素是对应的元素是谁，第二个元素是频次
votes.most_common(1)

# 预测值
predict_y = votes.most_common(1)[0][0] 
predict_y