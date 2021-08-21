---
layout: post
title: "假设检验"
date: 2021-08-20
description: "假设检验"
tag: 数据分析
katex: true
---

假设检验是利用样本信息去推断总体参数或分布的一种数据分析方法。

**假设检验的一般步骤：**

1. 提出原假设与备择假设
2. 给定显著性水平$\alpha$，选择合适的检验统计量，并确定其分布
3. 由$P(拒绝H_0 \mid H_0为真)=\alpha$确定$H_0$的拒绝域的形式
4. 由样本值求得检验统计量的观察值，若观察值在拒绝域内，则拒绝原假设$H_0$，否则在显著性水平$\alpha$下不能拒绝原假设

**假设检验的理论依据：**

1. 假设检验是概率意义下的反证法
2. 小概率事件不会发生

假设检验是在提出原假设$H_0$时，在样本上判定在原假设为真的情况下拒绝原假设的概率（p值），若小于显著性水平（可接受的最大拒真概率），则可以在显著性水平$\alpha$下拒绝原假设$H_0$。

**两类错误：**

| 研究结论\实际情况 |     $H_0$真     |     $H_0$假     |
| :---------------: | :-------------: | :-------------: |
|     拒绝$H_0$     | I类错误$\alpha$ |      正确       |
|   不能拒绝$H_0$   |      正确       | II类错误$\beta$ |

- $\alpha+\beta$不一定等于1
- 在样本量一定的情况下，$\alpha$与$\beta$不能同增同减
- 统计功效$Power=1-\beta$

## 参数检验

### 单一正态总体均值假设检验

#### 总体方差已知

假设正态总体$X \sim N(\mu, \sigma^2)$，则样本均值$\bar X \sim N(\mu, \frac {\sigma^2} n)$，有

$$
\frac {\bar X - \mu} {\frac \sigma {\sqrt n}} \sim N(0, 1)
$$

于是检验统计量可以构建为：

$$
z = \frac {\bar X - \mu_0} {\frac \sigma {\sqrt n}}
$$

#### 总体方差未知

已知

$$
\frac {\bar X - \mu} {\frac \sigma {\sqrt n}} \sim N(0, 1)
$$

又由[该文](https://blanket58.github.io/2021/08/statistical-theory/)t分布小节推导可知

$$
\frac {(n-1)s^2}{\sigma^2} \sim \chi^2(n - 1)
$$

有

$$
\frac {\frac {\sqrt n (\bar X - \mu)} \sigma} {\sqrt{\frac{\frac{(n-1)s^2}{\sigma^2}}{n - 1}}} = \frac {\sqrt n (\bar X - \mu)} s \sim t(n-1)
$$

于是检验统计量可以构建为：

$$
t = \frac {\sqrt n (\bar X - \mu_0)} s \sim t(n-1)
$$

当样本量大于30时，t分布近似于标准正态分布，此时可用z检验代替t检验，所以说t检验适用于小样本实验。

#### 假设检验

总体方差已知与总体方差未知的置信区间形式类似，以总体方差已知为例：

- 双侧

$$
H_0: \mu = \mu_0 \\
H_1: \mu \neq \mu_0
$$

若$\mid z \mid > z_{\frac \alpha 2}$则拒绝原假设，总体均值在$1- \alpha$置信度下的置信区间为：

$$
[\bar X - z_{\frac \alpha 2} \frac \sigma {\sqrt n}, \bar X + z_{\frac \alpha 2} \frac \sigma {\sqrt n}]
$$

- 左单侧

$$
H_0: \mu \ge \mu_0 \\
H_1: \mu < \mu_0
$$

若$z < -z_{\alpha}$则拒绝原假设，总体均值在$1- \alpha$置信度下的置信区间为：

$$
[\bar X - z_{\alpha} \frac \sigma {\sqrt n}, +\infty)
$$

- 右单侧

$$
H_0: \mu \leq \mu_0 \\
H_1: \mu > \mu_0
$$

若$z > z_{\alpha}$则拒绝原假设，总体均值在$1- \alpha$置信度下的置信区间为：

$$
(-\infty, \bar X + z_{\alpha} \frac \sigma {\sqrt n}]
$$

### 双独立正态总体均值假设检验

#### 总体方差$\sigma_1^2$、$\sigma_2^2$已知

假设$X$与$Y$相互独立，且$X \sim N(\mu_1, \sigma_1^2)$，$Y \sim N(\mu_2, \sigma_2^2)$，

样本均值：

$$
\bar X \sim N(\mu_1, \frac {\sigma_1^2} {n_1}) \\
\bar Y \sim N(\mu_2, \frac {\sigma_2^2} {n_2})
$$

样本均值差：

$$
\bar X - \bar Y \sim N(\mu_1 - \mu_2, \frac {\sigma_1^2} {n_1} + \frac {\sigma_2^2} {n_2})
$$

于是检验统计量可以构建为：

$$
z = \frac {(\bar X - \bar Y) - (\mu_1 - \mu_2)} {\sqrt {\frac {\sigma_1^2} {n_1} + \frac {\sigma_2^2} {n_2}}} \sim N(0, 1)
$$

#### 总体方差$\sigma_1^2$、$\sigma_2^2$未知，但$\sigma_1^2 = \sigma_2^2$

已知

$$
\frac {(n_1-1)s_1^2}{\sigma^2} \sim \chi^2(n_1 - 1) \\
\frac {(n_2-1)s_2^2}{\sigma^2} \sim \chi^2(n_2 - 1)
$$

由卡方分布可加性得：

$$
\frac {(n_1-1)s_1^2}{\sigma^2} + \frac {(n_2-1)s_2^2}{\sigma^2} \sim \chi^2(n_1 + n_2 - 2)
$$

又有

$$
\frac {(\bar X - \bar Y) - (\mu_1 - \mu_2)} {\sqrt {\frac {\sigma^2} {n_1} + \frac {\sigma^2} {n_2}}} \sim N(0, 1)
$$

于是检验统计量可以构建为：

$$
t = \frac {(\bar X - \bar Y) - (\mu_1 - \mu_2)} {\sqrt {\frac {\sigma^2} {n_1} + \frac {\sigma^2} {n_2}}} / \sqrt {\frac {\frac {(n_1-1)s_1^2}{\sigma^2} + \frac {(n_2-1)s_2^2}{\sigma^2}} {n_1 + n_2 - 2}} = \frac {(\bar X - \bar Y) - (\mu_1 - \mu_2)} {s_{pool} \sqrt {\frac 1 {n_1} + \frac 1 {n_2}}} \sim t(n_1 + n_2 - 2)
$$

其中：

$$
s_{pool} = \sqrt {\frac {(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2} {n_1 + n_2 - 2}}
$$

#### 总体方差$\sigma_1^2$、$\sigma_2^2$未知



### 两配对样本t检验



### 总体比率假设检验




## 非参数检验
