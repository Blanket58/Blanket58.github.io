---
layout: post
title: 评分卡建模全流程
date: 2021-12-14
tag: 风控
katex: true
mermaid: true
---

## 前言

我们需要认识到信用评分的发展是以风险评量为出发点，为求主题明确，聚焦于风险因子的探讨，**收益面并非考虑重点**。因此，风险极低者未必是获利贡献度最高的客户。同时，信用评分模型以“户数”为单位计算好坏比，我们可借此得知各分数下可能逾期的户数比率，而一般评估风险所惯用的逾期比率则是以“金额”为计算单位，两者计算基础不同，户数逾期率与金额逾期率无法相互比较。

根据使用时机，可以将信用评分卡分为三类：

- 申请评分（Application Score）
- 行为评分（Behavior Score）
- 催收评分（Collection Score）

## 构建流程

<div class="mermaid">
graph LR
    确定评分目的 --> 基本定义 --> 资料准备 --> 变量分析 --> 建立模型 --> 拒绝推论 --> 效力验证
</div>

### 基本定义

#### 观察期与表现期

观察期即为变量计算的历史区间。比如，有一变量为“近6个月逾期一期以上（M1+）的次数”，其观察期就等于6个月。观察期设定太长，可能无法反映近期状况，设定太短则稳定性不高，因此多半为6~24个月。

表现期则是准备预测的时间长度。例如，若想要预测客户未来12个月内违约的概率，则表现期为12个月。根据各种产品特性不同，表现期可能不同，通常设定为12~24个月。

#### 违约（Bad）定义

评分模型的任务在于区隔好坏客户，因此必须定义违约条件，这些条件并不限定为逾期，只要认定此情况为“非目标客户”。

#### 不确定定义

在某些条件下的客户，其风险处于较为模糊的灰色地带，很难将其归为好客户或坏客户。为强化模型的区隔能力，不确定的客户不适合被纳入建模样本中，不过在模型完成后可加入测试，观察其分数落点，理论上应以中等分数居多。在实际应用中，可利用转移分析（Roll Rate Analysis）观察各条件下的客户经过一段时间后的表现，以评估违约定义的区隔能力与稳定度，作为其选择好坏及不确定条件的参考。

![](/assets/2021-12-14-scorecard-1.png)

上表中B表示违约定义，I表示不确定定义，G表现正常定义。经过12个月的观察，原违约者大多数停留在违约状态，而原正常者转坏的比例也不高，这表示好坏客户的定义可被接受。原I03客户在12个月后明显往正常方向移动，因此可考虑将其改入正常定义组。

#### 评分范围

虽然信用评分可快速预测潜在风险，但并非所有状况都必须依赖评分来判断风险。如数据遗漏严重、数据期间过短和近来无信用往来记录者等状况的出现使这些客户的信息不足，对其评分也没有太大意义。

#### 样本分组

为了获得最佳的预测效果，通常可以根据客群或产品特性做样本分组，分别开发评分卡。若受限于时间，权宜之计为共享一张评分卡的同时调整不同应用场景下的准驳临界点，不过效果可能较差。适度的样本分组有助于提高模型的预测效果，不过要避免过度使用，如果切割过细，不但后续评分卡维护困难，且建模样本不足反而会影响模型的预测能力与稳定性。

### 变量分析

变量的形态可分为连续变量和离散变量。首先，从所有数据中挑选或组合出可能影响风险的变量，这些一开始先挑出的变量群被称为长变量列表，由于数量较多，因此必须先检查这些变量之间的相关性。若变量间存在高度相关性，之后只要依预测能力及稳定度择一保留即可。接下来进行单因子分析，以检查各变量的预测强度。

![](/assets/2021-12-14-scorecard-2.png)

这里以一个连续型变量来做示例，一开始根据数字大小切分较细的组别。分组的原则为：

- 组间差异大，组内差异小；
- 分组占率不低于5%；
- 各组中必须同时有好样本与坏样本。

其中WoE（Weight of Evidence）称为证据权重，计算公式为：

$$
WoE = \ln (\frac {Good_i} {Good_T} / \frac {Bad_i} {Bad_T}) = \ln \frac {Good_i} {Good_T} - \ln \frac {Bad_i} {Bad_T}
$$

negprob大于posprob时，WoE为负数，绝对值越高，表示该区间好坏样本的区隔能力越高。各组之间WoE值差距应尽可能拉开并呈现出单调的趋势。

![](/assets/2021-12-14-scorecard-3.png)

另一个重要的指标为信息量（Information Value），计算公式为：

$$
IV = \sum_{i=1}^n (\frac {Good_i} {Good_T} - \frac {Bad_i} {Bad_T}) \times WoE_i
$$

它是每个区间上WoE的加权和，可用来表示变量预测能力的强度。

|   信息量    |             预测能力              |
| :---------: | :-------------------------------: |
|    <0.02    |              Useless              |
| [0.02, 0.1) |               Weak                |
| [0.1, 0.3)  |              Medium               |
| [0.3, 0.5)  |              Strong               |
|    >=0.5    | Suspicious or too good to be true |

为了使IV提高，需要调整合并WoE相近的区间，最后得到的分组结果称为粗分类。待长变量列表中的所有变量IV都计算完成后，可从中挑选变量，优先排除高度相关、趋势异常、不容易解释的。经过筛选后的变量集合称为短变量列表，即为模型的候选变量。