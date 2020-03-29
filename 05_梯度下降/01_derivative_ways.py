#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:28:54 2020

@author: zhengsc
"""
# Sympy表达式求导
from sympy import *
# 符号化变量
x = Symbol('x')

func = 1/(1+x**2)

print("x:", type(x))
print("函数func: ",func)
print("func微分: ",diff(func, x))
print("代入点计算结果： ",diff(func, x).subs(x, 3))
print("结果转化为小数： ",diff(func, x).subs(x, 3).evalf())

# Scipy求导
# 使用次方法的时候需要将scipy降到1.2.1版本才能运行
from scipy import misc
def f(x):
    return x**3 + x**2

derivative(f, 1.0, dx=1e-6)

