# **KNN算法**
## 1 综述
### 1.1 knn算法的特点：
#### 1.1.1 优点
1. 简单朴素，容易理解。
2. 精确度高，对异常点不敏感。
3. 即可以做分类，也可以做回归、预测。
4. 解释性强。

#### 1.1.2 缺点
1. 从时间复杂度上看，容易发生纬度爆炸。
2. k的选择，需要借用交叉验证辅助
3. 训练效率低下，每次都需要遍历一遍计算距离。

### 1.2. knn的基本思想
k近邻不具有显式学习过程。选取距离样本最近的k个点，根据“少数服从多数”原则，以此来判断样本所属类别。
### 1.3 knn算法流程
- 三个要素：距离度量、k值、分类决策规则。
- 流程：
  -  计算测试对象到训练集中每个对象到距离。
  -  按距离远近进行排序。
  -  选取与测试对象最近的n个点，作为该对象到邻居。
  -  统计这k个点的频次。
  -  k个邻居中最高频次的类别即为测试对象的类别。
## 2 算法
### 2.1 sklearn库解读

```
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
           weights='uniform')
```
**参数说明：**
- **algorithm：** int，可选参数（默认值为5）。用于kneighbors查询的默认邻居的数量。
- **weights：** str or callabel（自定义类型），可选参数（默认值为‘uniform’）。用于预测的权重参数。
  -  **uniform：** 统一的权重。
  -  **distance：** 权重点等于他们距离的倒数。距离越近，权重越大。
  -  **[callable]：** 用户自定义的方法，此方法接收一个距离的数组，然后返回一个相同形状并且包含权重的数组。
- **algorithm:** {'auto','ball_tree', 'kd_tree','brute'},可选参数（默认为 'auto'）。计算最近邻居用的算法。
  - **ball_tree:** 使用算法ball_tree。
  - **kd_tree:** 使用算法KDTree。
  - **brute：** 使用暴力搜索。
  - **auto：** 会基于传入fit方法的内容，选择最合适的算法。注意 : 如果传入fit方法的输入是稀疏的，将会重载参数设置，直接使用暴力搜索。
- **leaf_size：** int，可选参数（默认值为30）。传入BallTree或者KDTree算法的叶子数量。此参数会影响构建、查询BallTree或者KDTree的速度，以及存储BallTree或者KDTree所需要的内存大小。此可选参数根据是否是问题所需选择性使用。
- **p:** integer, 可选参数(默认为 2)。用于Minkowski metric（闵可夫斯基空间）的超参数。p = 1, 相当于使用曼哈顿距离，p = 2, 相当于使用欧几里得距离]，对于任何 p ，使用的是闵可夫斯基空间。
- **metric：** string or callable, 默认为 ‘minkowski’。用于树的距离矩阵。默认为闵可夫斯基空间，如果和p=2一块使用相当于使用标准欧几里得矩阵。
- **metric_params：** dict, 可选参数(默认为 None)。给矩阵方法使用的其他的关键词参数。
- **n_jobs：** int, 可选参数(默认为 1)。用于搜索邻居的，可并行运行的任务数量。如果为-1, 任务数量设置为CPU核的数量。不会影响fit。
### 2.2 KNeighborsClassifier方法
方法名 | 含义
:--:|:--:
```fit(x,y)``` | 拟合.
```get_params([deep])```  | 获取估值器的参数.
```neighbors([X,n_neighbors,return_distance])``` | 查找一个或几个点的K个邻居。
```kneighbors_graph([X,n_neighbors,mode])```| 计算在X数组中每个点的k邻居的（权重）图。
```predict(X)```| 给提供的数据预测对应的标签。
```predict_proba(X)	```| 返回测试数据X的概率估值。
```score(X,y[,sample_weight])```| 返回给定测试数据和标签的平均准确值。
```set_params(**params)``` | 设置估值器的参数。

### 2.3 代码实现
**算法实现:** 参考knn_code.py <br />
**代码封装：** 参考KNN.py <br />
**sklearn实现：** 参考knn_from_sklearn.py