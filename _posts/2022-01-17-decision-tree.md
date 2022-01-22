---
layout: post
title: 决策树
date: 2022-01-17
tag: Algorithms
katex: true
---

## 基本流程

一般的，一棵决策树包含一个根结点、若干个内部结点和若干个叶结点；叶结点对应于决策结果，其他每个结点则对应于一个属性测试，每个结点包含的样本集合根据属性测试的结果被划分到子结点中；根结点包含样本全集。从根结点到每个叶结点的路径对应了一个判定测试序列。决策树学习的目的是为了产生一棵泛化能力强的决策树。

------

**输入：**

训练集$D = {(x_1, y_1),(x_2, y_2),...,(x_m, y_m)}$；属性集$A = {a_1, a_2, ..., a_d}$

**过程：**函数$TreeGenerate(D, A)$

1. 生成结点node;
2. if D中样本全属于同一类别C then
3. &nbsp;&nbsp;&nbsp;&nbsp;将node标记为C类叶结点; return
4. end if
5. if A = $\emptyset$ or D中样本在A上取值相同 then
6. &nbsp;&nbsp;&nbsp;&nbsp;将node标记为叶结点，其类别标记为D中样本数最多的类; return
7. end if
8. 从A中选择最优划分属性$a_{\ast}$;
9. for $a_{\ast}$中的每一个$a_{\ast}^v$ do
10. &nbsp;&nbsp;&nbsp;&nbsp;为node生成一个分支; 令$D_v$表示D中在$a_{\ast}$上取值为$a_{\ast}^v$的样本子集;
11. &nbsp;&nbsp;&nbsp;&nbsp;if $D_v$为空 then
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将分支结点标记为叶结点，其类别标记为D中样本最多的类; return
13. &nbsp;&nbsp;&nbsp;&nbsp;else
14. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以$TreeGenerate(D_v, A \setminus {a_{\ast}})$为分支结点
15. &nbsp;&nbsp;&nbsp;&nbsp;end if
16. end for

**输出：**以node为根节点的一棵决策树

------

决策树的生成是一个递归过程。在决策树的基本算法中，有三种情形会导致递归返回：

- 当前结点包含的样本全属于同一类别，无需划分；
- 当前属性集为空，或所有样本在所有属性上取值相同，无法划分；
- 当前结点包含的样本集合为空，不能划分。

第二种情况下，我们把该叶结点的类别设定为该结点所含样本最多的类别；第三种情况下，将该结点类别设置为其父结点所含样本最多的类别。前者使用当前结点的后验分布，后者使用当前结点的父结点的先验分布。

## 划分选择

选择最优划分属性的这一步很关键，随着划分过程不断进行，我们希望决策树的分支结点所包含的样本尽可能属于同一类别。

### 信息增益

假设当前样本集合D中第k类样本所占的比例为$p_k (k = 1, 2, ..., \vert \gamma \vert)$，则D的信息熵定义为：

$$
Ent(D) = - \sum_{k=1}^{\vert \gamma \vert} p_k \log_2{p_k}
$$

$Ent(D)$的值越小，则D的纯度越高。

假定离散属性a有V个可能的取值$\{a^1, a^2, ..., a^V\}$，若使用a来对样本集D进行划分，则会产生V个分支结点，其中第v个分支结点包含了D中所有在属性a上取值为$a^v$的样本，记为$D^v$。于是$D^v$的信息熵可以计算出来，再考虑到不同的分支结点所包含的样本数不同，给分支结点赋予权重$\vert D^v \vert / \vert D \vert$，即样本数越多的分支结点的影响越大，于是使用属性a对样本集D进行划分所获得的信息增益为：

$$
Gain(D, a) = Ent(D) - \sum_{v=1}^V \frac {\vert D^v \vert} {\vert D \vert} Ent(D^v)
$$

信息增益越大，使用属性a进行划分所获得的结点纯度提升越大。因此选择最优化分属性时的准则为：

$$
a_{\ast} = \mathop{\arg \max}\limits_{a \in A} \quad Gain(D, a)
$$

ID3决策树学习算法就使用信息增益准则来选择划分属性。

### 信息增益率

信息增益准则对可取值数目较多的属性有所偏好，属性取值越多，进行划分后每种取值下的分支结点纯度越高，但这会导致树的过拟合。C4.5决策树算法引入了信息增益率：

$$
Gain.ratio(D, a) = \frac {Gain(D, a)} {IV(a)}
$$

其中

$$
IV(a) = - \sum_{v=1}^{V} \frac {\vert D^v \vert} {\vert D \vert} \log_2{\frac {\vert D^v \vert} {\vert D \vert}}
$$

称为属性a的固有值（Intrinsic value）。属性a的可取值数目越多，则$IV(a)$越大。这样同时带来了一个问题，信息增益率准则会偏好取值数目较少的属性。因此C4.5采用了一个启发式算法：先从候选属性中找出**信息增益**高于平均水平的属性，再从中选择**增益率**最高的。

### 基尼系数

CART决策树使用基尼系数：

$$
\begin{aligned}
Gini(D) &= \sum_{k=1}^{\vert \gamma \vert} \sum_{k^{'} \neq k} p_k p_{k^{'}} \\
&= 1 - \sum_{k=1}^{\vert \gamma \vert} p_k^2
\end{aligned}
$$

基尼系数反映了从数据集D中随机抽取两个样本，其类别标记不一致的概率，因此越小结点纯度越高。同样的，属性a的基尼系数定义为：

$$
Gini.index(D, a) = \sum_{v=1}^V \frac {\vert D^v \vert} {\vert D \vert} Gini(D^v)
$$

因此选择最优化分属性时的准则为：

$$
a_{\ast} = \mathop{\arg \min}\limits_{a \in A} \quad Gini.index(D, a)
$$

## 剪枝处理

剪枝是决策树应对过拟合的主要手段。

**预剪枝：**

*在决策树生成过程中，对每个结点在划分前先进行估计，若当前划分不能带来决策树泛化性能的提升，则标记为叶结点。*预剪枝降低了过拟合的风险，还显著减少了决策树的训练时间开销和测试时间开销。但另一方面，有些分支的当前划分虽不能提升泛化性能，甚至可能导致泛化性能暂时下降，但在其基础上进行的后续划分却有可能导致性能显著提高，预剪枝基于贪心原则禁止这些分支的展开，使得预剪枝决策树可能会欠拟合。

**后剪枝：**

*先从训练集生成一棵完整的决策树，然后由底向上，判断若当前非叶结点标记为叶结点，能否带来决策树泛化性能的提升，若可以则将该结点替换为叶结点。*后剪枝决策树通常会比预剪枝决策树保留更多的分支，一般情况下后剪枝决策树的欠拟合风险小，泛化性能往往由于预剪枝决策树，但后剪枝过程需要先生成一棵完整的决策树，再自底向上对每一个非叶结点进行逐一考察，训练时间开销要大得多。

## 连续与缺失值

### 连续值处理

上面讨论的都是基于离散属性的决策树，对于连续型属性，需要使用连续属性离散化技术，C4.5采用了**二分法**。

给定样本集D和连续属性a，假定a在D上出现了n个不同的取值，将这些值从小到大进行排序，记为$\{a^1, a^2, ..., a^n\}$。基于划分点t可将D分为子集$D_t^-$和$D_t^+$，其中$D_t^-$包含那些在属性a上取值小于等于t的样本。对于相邻的属性取值$a^i$和$a^{i+1}$来说，t在区间$[a^i, a^{i+1})$中取任意值所产生的划分结果相同。因此对于连续属性a我们可考察包含n-1个元素的候选划分点集合：

$$
T_a = \{\frac {a^i + a^{i+1}} 2 \vert 1 \leq i \leq n - 1 \}
$$

即把区间的中位点当作候选划分点，于是我们就可以像离散属性一样来考察这些划分点。

$$
\begin{aligned}
Gain(D, a) &= \max_{t \in T_a} Gain(D, a, t) \\
&= \max_{t \in T_a} Ent(D) - \sum_{\lambda \in \{-, +\}} \frac {\vert D_t^{\lambda} \vert} {\vert D \vert} Ent(D_t^{\lambda})
\end{aligned}
$$

不同于离散属性，若当前结点划分属性为连续属性，该属性还可作为其后代结点的划分属性。

### 缺失值处理

现实情况中经常会存在样本的某些属性值缺失，C4.5采用了如下的解决方法。

#### 如何在属性值缺失的情况下进行划分属性选择？

给定样本集D和连续属性a，令$\tilde D$表示D中在属性a上没有缺失值的样本子集。假定属性a有V个可取值$\{a^1, a^2, ..., a^V\}$，令$\tilde D^v$表示$\tilde D$中在属性a上取值为$a^v$的样本子集，$\tilde D_k$表示$\tilde D$中属于第k类（$k = 1, 2, ..., \vert \gamma \vert$）的样本子集，于是就有$\tilde D = \bigcup_{k=1}^{\vert \gamma \vert} \tilde D_k$，$\tilde D = \bigcup_{v=1}^{V} \tilde D^v$。假定为每个样本x赋予一个权重$w_x$，并定义

$$
\rho = \frac {\sum_{x \in \tilde D} w_x} {\sum_{x \in D} w_x} \\
\tilde p_k = \frac {\sum_{x \in \tilde D_k} w_x} {\sum_{x \in \tilde D} w_x} \quad (1 \leq k \leq \vert \gamma \vert) \\
\tilde r_v = \frac {\sum_{x \in \tilde D^v} w_x} {\sum_{x \in \tilde D} w_x} \quad (1 \leq v \leq V)
$$

其中对属性a，$\rho$表示无缺失值样本所占的比例，$\tilde p_k$表示无缺失值样本中第k类所占的比例，$\tilde r_v$表示无缺失值样本中在属性a取值$a^v$的样本所占的比例。显然$\sum_{k = 1}^{\vert \gamma \vert} \tilde p_k = 1$，$\sum_{v = 1}^{V} \tilde r_v = 1$。于是信息增益的计算公式可以推广为：

$$
\begin{aligned}
Gain(D, a) &= \rho \times Gain(\tilde D, a) \\
&= \rho \times (Ent(\tilde D) - \sum_{v=1}^{V} Ent(\tilde D^v))
\end{aligned}
$$

同时

$$
Ent(\tilde D) = - \sum_{k=1}^{\vert \gamma \vert} \tilde p_k \log_2{\tilde p_k}
$$

#### 给定划分属性，若样本在该属性上的值缺失，如何对样本进行划分？

若样本x在划分属性a上的取值已知，则将x划入与其取值对应的子结点，且样本权重维持$w_x$不变；若样本x在划分属性a上的取值未知，则将x同时划入所有子结点，并将样本权重在与属性值$a^v$对应的子结点中调整为$\tilde r_v \cdot w_x$。直观地看，这就是让同一个样本以不同的概率划入到不同的子结点中去。

## 多变量决策树

若把每个属性都视为坐标空间中的一个坐标轴，则d个属性描述的样本就对应了d维空间中的一个数据点，对样本分类就意味着在这个坐标空间中寻找不同类样本之间的分类边界。决策树所形成的分类边界有一个明显的特点就是**轴平行**，它的分类边界由若干个与坐标轴平行的分段组成。但真实数据上的分类边界往往很复杂，需要很多段划分才能获得较好的近似，时间开销将会很大。如果能使用“斜”的划分边界，决策树模型将大为简化。

多变量决策树，非叶节点不再仅对某个属性，而是对属性的线性组合进行测试。在多变量决策树的学习过程中，不是为每个非叶节点寻找一个最优划分属性，而是试图建立一个合适的线性分类器。

## 示例

绘制CART分类树的详细分支结果。

```python
import pydotplus
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
dot_data = export_graphviz(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
pydotplus.graph_from_dot_data(dot_data).write_png('iris.png')
```

![](/assets/2022-01-17-decision-tree-1.png)

## 参考文献

[1] 周志华.机器学习[M].北京：清华大学出版社，2016：73-95

[2] [简单Python决策树可视化实例](https://blog.csdn.net/u012845311/article/details/77294973)
