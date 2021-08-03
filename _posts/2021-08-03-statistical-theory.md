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

\begin{equation}
\bar{X} = \frac{1}{n} \sum X_i
\end{equation}

### 样本方差

\begin{equation}
s^2 = \frac{1}{n-1} \sum (X_i - \bar{X})^2
\end{equation}

### 变异系数

\begin{equation}
CV = \frac{s}{\bar{X}}
\end{equation}

样本标准差除以样本均值。当比较两组不同量纲数据的离散程度时，直接用标准差比较不合适，应该消除量纲的影响。

### 异众比率

\begin{equation}
V_r = 1 - \frac{f_m}{\sum f_i}
\end{equation}

其中$f_m$是众数组的频数，$\sum f_i$是总频数。

### k阶原点矩

\begin{equation}
m_k = \frac{1}{n} \sum X^k_i
\end{equation}

### k阶中心距

\begin{equation}
v_k = \frac{1}{n-1} \sum (X_i - \bar{X})^k
\end{equation}

### 偏度

\begin{equation}
skew = E[(\frac{X - \mu}{\sigma})^3]
\end{equation}

即3阶中心距。

![](/assets/2021-08-03-statistical-theory-1.png)

### 峰度

\begin{equation}
kurt = E[(\frac{X - \mu}{\sigma})^4]
\end{equation}

即4阶中心距。

![](/assets/2021-08-03-statistical-theory-2.png)

## 分布

### 二项分布

\begin{equation}
B(n, p) \\\\
EX = np, DX = np(1 - p)
\end{equation}

### 泊松分布

\begin{equation}
P(x) = \frac{\lambda^x e^{-\lambda}}{x!}, x = 0, 1, 2, ... \\\\
EX = \lambda, DX = \lambda
\end{equation}

### 正态分布

\begin{equation}
f(x) = \frac 1 {\sqrt{2\pi} \sigma} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}, x \in \mathbb{R} \\\\
EX = \mu, DX = \sigma^2
\end{equation}

### 指数分布

\begin{equation}
f(x) = \left\{
\begin{array}{l}
\lambda e^{-\lambda x}, x>0 \\\\
0, x \le 0
\end{array}
\right.
\end{equation}

或

\begin{equation}
f(x) = \left\{
\begin{array}{l}
\frac 1 \theta e^{-\frac x \theta}, x>0 \\\\
0, x \le 0
\end{array}
\right.,
\theta = \frac 1 \lambda
\end{equation}

\begin{equation}
F(x) = \left\{
\begin{array}{l}
1 - e^{-\lambda x}, x>0 \\\\
0, x \le 0
\end{array}
\right.
\end{equation}

\begin{equation}
EX = \frac 1 \lambda, DX = \frac 1 \lambda^2
\end{equation}

### 卡方分布

若$X_1, ..., X_n$相互独立，且$X_i \sim N(0, 1)$，则

\begin{equation}
\sum^n_{i=1} \sim \chi^2 (n) \\\\
EX = n, DX = 2n
\end{equation}

### t分布

\begin{equation}
EX = 0, DX = \frac n {n - 2}
\end{equation}

若$X \sim N(0, 1)$，$Y \sim \chi^2(n)$，$X$与$Y$相互独立，则

\begin{equation}
t = \frac X {\sqrt{\frac Y n}}
\end{equation}

若$X \sim N(\mu, \sigma^2)$，则

\begin{equation}
\bar X = \frac 1 n \sum^n_{i=1} X_i \\\\
E \bar X = \frac 1 n \sum EX_i = \mu \\\\
D \bar X = \frac 1 {n^2} \sum EX_i = \frac {\sigma^2} n
\end{equation}

因此：

\begin{equation}
\frac {\bar X - \mu}{\frac \sigma {\sqrt n}} \sim N(0, 1)
\end{equation}

又有如下的推导存在：

\begin{equation}
s^2 = \frac 1 {n - 1} \sum^n_{i=1} (X_i - \bar X)^2 \\\\
\begin{aligned}
\frac {(n-1)s^2}{\sigma^2} &= \frac {\sum(X_i - \bar X)^2}{\sigma^2} \\\\
&= \sum (\frac{X_i}{\sigma} - \frac{\bar X}{\sigma})^2 \\\\
&= \sum (\frac{X_i - \mu}{\sigma} - \frac{\bar X - \mu}{\sigma})^2 \\\\
&= \sum (Z_i - \bar Z)^2 \\\\
&= \sum (Z_i^2 - 2Z_i \bar Z + \bar Z^2) \\\\
&= \sum^n_{i=1} Z^2_i - n\bar Z^2 \sim \chi^2(n - 1)
\end{aligned}
\end{equation}

故

\begin{equation}
\frac {\frac {\sqrt n (\bar X - \mu)} \sigma} {\sqrt{\frac{\frac{(n-1)\s^2}{\sigma^2}}{n - 1}}} = \frac {\sqrt n (\bar X - \mu)} s \sim t(n-1)
\end{equation}

