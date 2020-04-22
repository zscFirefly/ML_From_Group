#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:45:27 2020

@author: a111
"""


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 绘制图片方法
def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                             subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.1, wspace=0.1)) 
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
    plt.show() 


digits = datasets.load_digits()
X = digits.data
y = digits.target
# 添加一个正态分布的噪音矩阵
noisy_digits = X + np.random.normal(0, 4, size=X.shape)
# 绘制噪音数据：从y==0数字中取10个，进行10次循环；
# 依次从y==num再取出10个，将其与原来的样本拼在一起
example_digits = noisy_digits[y==0,:][:10]
for num in range(1,10):
    example_digits = np.vstack([example_digits, noisy_digits[y==num,:][:10]])

print("测试样本大小",example_digits.shape)
plot_digits(example_digits)


pca = PCA(0.5).fit(noisy_digits)
print("输出主成分个数",pca.n_components_)
# 输出：12，即原始数据保存了50%的信息，需要12维
# 进行降维、还原过程
components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)