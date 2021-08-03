---
layout: post
title: "一些统计知识点"
date: 2021-08-03
description: "一些统计知识点"
tag: 数据分析
mathjax: true
---

## 统计量

设$X_1, ..., X_n$是从总体$X$中抽样的样本，如果由此样本构造一个不依赖任何未知参数的函数$T(X_1, ..., X_n)$，则称函数$T$是一个统计量。

### 样本均值

$$
\overline{X} = \frac{1}{n} \sum X_i
$$

### 样本方差

$$
s^2 = \frac{1}{n-1} \sum (X_i - \bar{X})^2
$$

### 变异系数

$$
CV = \frac{s}{\bar{X}}
$$

样本标准差除以样本均值。当比较两组不同量纲数据的离散程度时，直接用标准差比较不合适，应该消除量纲的影响。

### 异众比率

$$
V_r = 1 - \frac{f_m}{\sum f_i}
$$

其中$f_m$是众数组的频数，$\sum f_i$是总频数。

### k阶原点矩

$$
m_k = \frac{1}{n} \sum X^k_i
$$

### k阶中心距

$$
v_k = \frac{1}{n-1} \sum (X_i - \bar{X})^k
$$

### 偏度

$$
skew = E[(\frac{X - \mu}{\sigma})^3]
$$

即3阶中心距。

![](/assets/2021-08-03-statistical-theory-1.png)

### 峰度

$$
kurt = E[(\frac{X - \mu}{\sigma})^4]
$$

即4阶中心距。

![](/assets/2021-08-03-statistical-theory-2.png)

## 分布

