---
layout: post
title: "概率论与数理统计"
date: 2021-08-03
description: "概率论与数理统计"
tag: 数据分析
katex: true
---

## 统计量

设$X_1, ..., X_n$是从总体$X$中抽样的样本，如果由此样本构造一个不依赖任何未知参数的函数$T(X_1, ..., X_n)$，则称函数$T$是一个统计量。

### 样本均值

$$
\bar{X} = \frac{1}{n} \sum X_i
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

### 二项分布

$$
B(n, p) \\
EX = np, DX = np(1 - p)
$$

### 泊松分布

$$
P(x) = \frac{\lambda^x e^{-\lambda}}{x!}, x = 0, 1, 2, ... \\
EX = \lambda, DX = \lambda
$$

### 正态分布

$$
f(x) = \frac 1 {\sqrt{2\pi} \sigma} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}, x \in \mathbb{R} \\
EX = \mu, DX = \sigma^2
$$

### 指数分布

$$
f(x) =
\begin{cases}
\lambda e^{-\lambda x}, x>0 \\
0, x \le 0
\end{cases}
$$

或

$$
f(x) =
\begin{cases}
\frac 1 \theta e^{-\frac x \theta}, x>0 \\
0, x \le 0
\end{cases}
\quad \theta = \frac 1 \lambda
$$

$$
F(x) =
\begin{cases}
1 - e^{-\lambda x}, x>0 \\
0, x \le 0
\end{cases}
$$

$$
EX = \frac 1 \lambda, DX = \frac 1 {\lambda^2}
$$

### 卡方分布

若$X_1, ..., X_n$相互独立，且$X_i \sim N(0, 1)$，则

$$
\sum^n_{i=1} X^2_i \sim \chi^2 (n) \\
EX = n, DX = 2n
$$

### t分布

$$
EX = 0, DX = \frac n {n - 2}
$$

若$X \sim N(0, 1)$，$Y \sim \chi^2(n)$，$X$与$Y$相互独立，则

$$
t = \frac X {\sqrt{\frac Y n}}
$$

若$X \sim N(\mu, \sigma^2)$，则

$$
\bar X = \frac 1 n \sum^n_{i=1} X_i \\
E \bar X = \frac 1 n \sum EX_i = \mu \\
D \bar X = \frac 1 {n^2} \sum DX_i = \frac {\sigma^2} n
$$

因此：

$$
\frac {\bar X - \mu}{\frac \sigma {\sqrt n}} \sim N(0, 1)
$$

又有如下的推导存在：

$$
s^2 = \frac 1 {n - 1} \sum^n_{i=1} (X_i - \bar X)^2 \\
\begin{aligned}
\frac {(n-1)s^2}{\sigma^2} &= \frac {\sum(X_i - \bar X)^2}{\sigma^2} = \sum (\frac{X_i}{\sigma} - \frac{\bar X}{\sigma})^2 \\
&= \sum (\frac{X_i - \mu}{\sigma} - \frac{\bar X - \mu}{\sigma})^2 \\
&= \sum (Z_i - \bar Z)^2 \\
&= \sum (Z_i^2 - 2Z_i \bar Z + \bar Z^2) \\
&= \sum^n_{i=1} Z^2_i - n\bar Z^2 \sim \chi^2(n - 1)
\end{aligned}
$$

故

$$
\frac {\frac {\sqrt n (\bar X - \mu)} \sigma} {\sqrt{\frac{\frac{(n-1)s^2}{\sigma^2}}{n - 1}}} = \frac {\sqrt n (\bar X - \mu)} s \sim t(n-1)
$$

### F分布

若$Y$与$Z$相互独立，$Y \sim \chi^2(m)$，$Z \sim \chi^2(n)$，则

$$
X = \frac {\frac Y m}{\frac Z m} \sim F(m, n) \\
EX = \frac n {n - 2} \\
DX = \frac {2 n^2(m+n-2)} {m(n-2)(n-4)}, n>4
$$

## 定理

### 中心极限定理

从均值为$\mu$，方差为$\sigma^2$的任意总体中抽取样本量为$n$的样本，当$n$充分大时，样本均值$\bar X$的抽样分布近似服从正态分布$N(\mu, \frac {\sigma^2} n)$。

### 大数定理

