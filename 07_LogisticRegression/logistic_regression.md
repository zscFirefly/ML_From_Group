# **逻辑回归**
## 1 综述
### 1.1 probit回归
假设场景：银行理财产品，判断用户是否购买。
已知用户数据`$X=\{x_{1},x_{2},x_{3},x_{4}...\}$`。
设置用户是否购买该产品的正负效用为`$y^{*},\bar y$`。记用户的购买行为为`$y$`。于是有：
```math
y^{*}=f(X),\bar y=G(X)
```
```math
y = \begin{cases} 
1 & y^{*}>\bar y \\
0 & y^{*}<\bar y
\end{cases}
```
对`$y^{*},\bar y$`使用回归，有如下方程
```math
y^{*} = X_{i}\varphi+\theta_{i}\\
\bar y = X_{i}\omega+\tau_{i}
```
其中`$\theta_{i},\tau_{i}$`都是相互独立的随机变量且服从正态分布。\
令：
```math
z_{i} = y^{*}-y\\
\gamma = \varphi - \omega\\
\epsilon = \theta - \tau
```
于是可以得到`$z=X\gamma+\epsilon$`,问题转化为以下阶梯函数
```math
y = \begin{cases} 
1 & X\gamma+\epsilon > 0 \\
0 & X\gamma+\epsilon < 0
\end{cases}
```
**转化为概率问题**有：
```math
\begin{aligned}
P(y=1)  &= P(X\gamma+\epsilon > 0)\\
        &= P(\epsilon > -X\gamma)\\
        &= 1 - P(\epsilon <= -X\gamma)\\
        &= 1 - F_{\epsilon}(-X\gamma)\\
\end{aligned}
```
其中：**`$F_{\epsilon}$`是随机变量`$\epsilon$`的累积分布函数**。\
上述模型称为**probit回归**。在模型过程中假设了正负效用变量`$y^{*},\bar y$`，该变量被成为**隐藏变量**。\
**对于一个分类问题，由于“窗口效用”，我们只能看见客户的购买行为，但是在分类的背后，是隐藏变量之间的博弈，我们通过搭建隐藏变量的模型，来求出客户购买的概率。**
### 1.2 sigmoid函数
标准逻辑分布的概率密度函数
```math
f(x)=\frac {e^{-x}} {(1+e^{-x})^{2}}
```
累计分布函数
```math
\sigma(t)=\frac {1} {(1+e^{-t})}
```
将上节所讲的**正负效用函**数之差带入到sigmoid函数可以得到
```math
\hat p = \sigma(\theta^{T}X_{b})=\frac{1}{(1+e^{-\theta^{T}X_{b}})}\\

\hat y =  \begin{cases} 
1 & \hat p > 0.5 \\
0 & \hat p < 0.5
\end{cases}
```
上述模型即为**逻辑回归**。
### 1.3 逻辑回归
- 逻辑回归是什么？
    - 逻辑回归假设数据服从伯努利分布，利用极大似然函数的方法，运用梯度下降来求解参数，从而实现而分类效果。
#### 1.3.1 逻辑回归损失函数
##### 1.3.1.1 推导构造
逻辑回归是根据逻辑回归本身式子中系数的最大似然估计推导而来的。
令逻辑回归的模型为`$h_{0}(x;\theta)$`。可以将其视为类1的后验概率。于是有：
```math
p(y=1|x;\theta)=\frac{1}{1+e^{-\theta^{T}X_{b}}}\\
p(y=0|x;\theta)=1-\frac{1}{1+e^{-\theta^{T}X_{b}}}
= \frac{e^{-\theta^{T}X_{b}}}{1+e^{-\theta^{T}X_{b}}}
```
将上述式子改写成一般式有：
```math
p(y|x;\theta)=h_{0}(x;\theta)^{y}(1-h_{0}(x;\theta))^{1-y}
```
根据极大似然估计有：
```math
J(\theta)=\prod_{i=1}^{m}p(y^{i}|x^{i};\theta)=h_{0}(x;\theta)^{y^{i}}(1-h_{0}(x;\theta))^{(1-y)^{i}}
```
为了简化计算，取对数有：
```math
log(J(\theta))=\sum_{i=1}^{m}{y^{(i)}}log(\hat{p}^{(i)})+(1-y^{(i)})log(1-\hat{p}^{(i)})
```
对于给定的样本m，我们希望损失函数`$\frac{1}{m}log(J(\theta))$`越小越好，于是有损失函数：
```math
J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}{y^{(i)}}log(\sigma(\theta^{T}X_{b}))+(1-y^{(i)})log(1-\sigma(\theta^{T}X_{b})))
```

##### 1.3.1.2 推导求解
由于
```math
\sigma(t)=\frac{1}{1+e^{-t}}
```
求其导有
```math
\begin{aligned}
\sigma^{'}(t)
&= -(1+e^{-t})^{-2}*-e^{-t} \\
&= \frac{e^{-t}}{(1+e^{-t})^{2}} \\
&= \frac{e^{-t}}{(1+e^{-t})}*\frac{1}{(1+e^{-t})}\\
&= (1-\sigma(t))*\sigma(t)\\
\end{aligned}
```
对`$log(\theta)$`求导有
```math
\begin{aligned}
(log(\sigma(t)))^{'}
&=\frac{1}{\sigma(t)}\sigma^{'}(t)\\
&=\frac{1}{\sigma(t)}(1-\sigma(t))*\sigma(t) \\
&=1-\sigma(t)
\end{aligned}
```
对`$y^{(i)}log(\sigma(\theta^{T}X_{b}^{(i)}))$`求导有
```math
\begin{aligned}
(y^{(i)}log(\sigma(\theta^{T}X_{b}^{(i)})))^{'}
&=y^{(i)}(1-\sigma(\theta^{T}X_{b}^{i}))X_{j}^{i}
\end{aligned}
```
同理对`$(1-y^{(i)})log(1-\sigma(\theta^{T}X_{b}))$`求导有。
```math
\begin{aligned}
((1-y^{(i)})log(1-\sigma(\theta^{T}X_{b})))^{'}
&=(1-y^{(i)})(-\sigma(\theta^{T}X_{b}^{i}))X_{j}^{i}
\end{aligned}
```
两者求和得最终梯度为
```math
\frac{\partial L(\theta)}{\partial \theta_{j}}
=\frac{1}{m}\sum_{i=1}^{m}(\sigma(\theta^{T}X_{b}^{(i)})-y^{(i)})X_{j}^{(i)}
```

### 1.4 决策边界
令
```math
\sigma(t)=\frac{1}{1+e^{-\theta^{T}x_{b}}}=0
```
解得
```math
\theta^{T}x_{b}=0
```
则决策平面为一条直线。
假设存在两个特征`$x_{1},x_{2}$`,带入有`$\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}=0$`，可化简为
```math
x2 = \frac{-\theta_{0}-\theta_{1}x_{1}}{\theta_{2}}
```
### 1.4 逻辑回归中对正则化
在逻辑回归中，相对与L1、L2正则，更常用的一种做法是在损失函数前加一个超参数C。即：`$C*J(\theta)+L1$`,该超参数相当于`$\alpha$`前的一个倒数。

## 2 代码实现
文件名 | 描述 
:-:|:-:
01_logistic_regression_mycode.py|逻辑回归手写代码
02_desicion_line.py|多项式逻辑回归与决策平面
03_overfit.py|过拟合下的正则测试

## 3 参考链接
出场率No.1的逻辑回归算法，是怎样“炼成”的？：https://mp.weixin.qq.com/s/xfteESh2bs1PTuO2q39tbQ\
逻辑回归的本质及其损失函数的推导、求解：https://mp.weixin.qq.com/s/nZoDjhqYcS4w2uGDtD1HFQ\
逻辑回归代码实现与调用：https://mp.weixin.qq.com/s/ex7PXW9ihr9fLOjlo3ks4w\
逻辑回归的决策边界及多项式:https://mp.weixin.qq.com/s/97CA-3KlOofJGaw9ukVq1A\
sklearn中的逻辑回归中及正则化:https://mp.weixin.qq.com/s/BUDdj4najgR0QAHhtu8X9A