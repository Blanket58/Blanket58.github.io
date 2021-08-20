---
layout: post
title: "ABTest样本量"
date: 2021-08-02
description: "ABTest样本量"
tag: 数据分析
katex: true
---

## 效应量

效应量是一个统计概念，它在数字尺度上衡量两个变量之间关系的强度。例如，如果我们有关于男性和女性身高的数据，并且我们注意到，平均而言，男性比女性高，男性身高和女性身高之间的差异被称为效应量。效应量越大，男女身高差越大。统计效应量帮助我们确定差异是真实的还是由于因素的变化。在假设检验中，效应量、统计功效、样本量和显著性水平彼此相关。在统计分析中，效应大小通常以三种方式衡量：

1. 标准均差(standardized mean difference)
2. 几率(odd ratio)
3. 相关系数(correlation coefficient)

### 皮尔森相关系数

根据Cohen的定义，相关系数在0.1附近时效应量低，在0.3附近时为中等水平，比0.5大时效应量大。
$$
r = \frac {\sum_{i=1}^n (X_i - \bar X) (Y_i - \bar Y)} {\sqrt{\sum_{i=1}^n (X_i - \bar X)^2} \sqrt{\sum_{i=1}^n (X_i - \bar X)^2}}
$$

### 标准化均差

$$
\theta = \frac {\mu_1 - \mu_2} \sigma
$$

### Cohen’s d

$$
d = \frac{\bar x_1 - \bar x_2} {\sqrt{\frac {(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2} {n_1 + n_2}}}
$$

### Glass’s $\Delta$

与Cohen’s d相似，但分母使用的是第二组的标准差。
$$
\Delta = \frac {\bar x_1 - \bar x_2} {s_2}
$$

### Hedge’s g

$$
g = \frac {\bar x_1 - \bar x_2} {s^*} \\

s^* = \sqrt{\frac {s_1^2(n_1 - 1)+s_2^2(n_2 - 1)} {n_1 + n_2 - 2}}
$$

### Cohen’s $f^2$

该效应量用来做方差分析时衡量多个回归之间的差距。
$$
f^2 = \frac {R^2} {1 - R^2}
$$

### Cramer’s $\phi$



### Cramer’s V



### Odd ratio
