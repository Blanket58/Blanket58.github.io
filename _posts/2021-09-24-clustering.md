---
layout: post
title: 聚类
date: 2021-09-24
tag: Algorithms
katex: true
---

## K-means

**原理：**

1. 首先随机从样本集中找K个点当作K个聚类簇的均值点
2. 计算所有样本点分别与各个均值点的距离，将样本点归入距离最小的簇中
3. 重新计算每个聚类簇的均值点位置
4. 重复操作，直至达到指定迭代次数、或临近两次迭代均值点的Frobenius范数变动小于阈值

> 其中Frobenius范数也称欧几里得范数，即矩阵中每个元素的平方和再开方。在这里矩阵指的是K行n列的质心矩阵，它的每一行为一个质心，每一列为该质心在各个维度上的坐标。
>

**优点：**

- 原理简单、实现容易、收敛速度快；
- 聚类效果较优；
- 算法的可解释性比较强；
- 主要需要调参的参数仅仅是簇数K。

**缺点：**

- K值的选取不好把握；
- 对于不是[凸的数据集](/2021/02/machine-learning/)比较难收敛；
- 如果各隐含类别的数据不平衡，比如各隐含类别的数据量严重失衡，或者各隐含类别的方差不同，则聚类效果不佳；
- 采用迭代方法，得到的结果只是局部最优。
