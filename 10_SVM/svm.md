# **支持向量机**
## 1 综述
### 1.1 支持向量机
- 什么是支持向量机？
    - 硬间隔最大化：通过支持向量，寻找一个最大边界的决策超平面。
    - 软间隔最大化：在硬间隔最大化上增加一个松弛因子。
- 推导：
    - 转化为最优化问题
    - 求解有条件限制的最优化问题
        - 有约束条件的最优化问题，用拉格朗日乘数法来解决。
        - 求导。
        - 转换对偶问题。
        - 求`$a、\omega、b$`

- 核函数（将低纬数据映射到高维数据上）
    - 多项式核函数
    - 高斯核函数

- 其他：
    - 可应用于回归
    - 数据使用需要标准化

- 优点：
    - 解决高维特征的分类问题和回归问题很有效。
    - 仅仅使用一部分支持向量来做超平面的决策，无需依赖全部数据。
    - 有大量的核函数可以使用，从而可以很灵活的来解决各种非线性的分类回归问题。
    - 样本量不是海量数据的时候，分类准确率高，泛化能力强。
- 缺点：
    - 样本量不是海量数据的时候，分类准确率高，泛化能力强。
    - SVM在样本量非常大，核函数映射维度非常高时，计算量过大，不太适合使用。
    - 非线性问题的核函数的选择没有通用标准，难以选择一个合适的核函数。
    - SVM对缺失数据敏感。
- 总结：
    - 最简单的情形Linear SVM。
    - 实际情况下不能完美地将各个数据点分开，存在噪声数据。为了解决线性该问题，就提出了Soft Margin的思想，增加正则化项，提高容错性。
    - 为了解决线性不可能问题呢？把原始空间的问题映射到高维空间，将线性不可分问题转换成线性可分问题，即为核函数。


## 2 代码实践
### 2.1 代码文件
文件名 | 描述 
:-:|:-:
01_svm_linner_plot.py|线性可分svm绘图
02_svm_nolinner.py|非线性svm绘图
03_kernel_fuc.py|核函数回归
04_svm_regression.py|svm应用于回归

## 3 参考链接
入门支持向量机1：图文详解SVM原理与模型数学推导：https://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484477&idx=1&sn=226e099c1951b6c11b1e7fb6b7a092a3&chksm=eb932d8bdce4a49d0595b6c642fc2e5969fdc05a185f97a39cc1a896e24d56d8703541a28f9c&scene=21#wechat_redirect\
入门支持向量机2:软间隔与sklearn中的SVM: https://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484512&idx=1&sn=7a6b75f312e92bbdecafdedf979ed929&chksm=eb932dd6dce4a4c0ae4ea087878ec7a5f5ccc0724a85aa93daff3d08c33ecf86a3d809e51a82&scene=21#wechat_redirect\
入门支持向量机3：巧妙的Kernel Trick:https://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484546&idx=1&sn=33c6c5cb698b8835b2ee57dd8ea7c221&chksm=eb932d34dce4a4221f40f3daa26863a5fd05dcbcf74738d5423316c643e3ff930904d1a33fca&scene=21#wechat_redirect\
入门支持向量机4：多项式核函数与RBF核函数代码实现及调参:https://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484572&idx=1&sn=fd6e86ce45167286fb6ba4089b7b29dd&chksm=eb932d2adce4a43c44d26e79d4968f395d7cc22a31d84aef7944b227e1843b3f0722a5e894ed&scene=21#wechat_redirect \
入门支持向量机5：回归问题及系列回顾总结:https://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484596&idx=1&sn=7e93eb135d66c86238ccf516f0ae65ec&chksm=eb932d02dce4a41447a9cb34d627f435c760a5deb125a40d4c2a77f99e2187194d6bfbda4cbc&scene=21#wechat_redirect
