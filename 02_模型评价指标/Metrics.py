#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:03:21 2020

@author: reocar
"""
from math import sqrt
import numpy as np

def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0],  "the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)