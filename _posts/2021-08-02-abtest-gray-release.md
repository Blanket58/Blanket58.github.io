---
layout: post
title: "ABTest灰度上线的具体实现"
date: 2021-08-02
description: "ABTest灰度上线的具体实现"
tag: 数据分析
mermaid: true
---

与已知样本总量去切分比例不同，灰度上线时我们不知道用户总量会是多少，因此在线上如何实现具体的切分算法十分关键，这里介绍一种我在使用的方法。

## 理论

<div class="mermaid">
graph LR
    id1([userid])
    id2(prefix + userid)
    id3(hash为byte)
    id4(取前8字节)
    id5(转为整数)
    id6(8字节能表示的最大整数为18446744073709551615)
    id7(求除法将结果与阈值比较得出灰度分组)
    id1 --> id2 --> id3 --> id4 --> id5 --> id7
    id6 --> id7
</div>

如此的伪随机分配算法，可以实现实验的复现，同一个用户，无论多少次进入同一个实验，都被会分配进同一个组别。同时，通过给用户名前添加前缀，可以保证不同实验在切分时，同一个用户不会都被分进测试组中，造成客户体验的问题。

## Python实现

```python
from hashlib import sha256


def gray_release(prefix: str, key: str, low: float, high: float) -> bool:
    return low <= int(sha256((prefix + key).encode('ascii')).digest()[:8].hex(), 16) / (0xffffffffffffffff + 1) < high
```

下面模拟数据对该函数进行验证：

```python
from collections import Counter

x = ['A' if gray_release('test', str(x), 0, .5) else 'B' for x in range(10000)]
Counter(x)
# Counter({'A': 5025, 'B': 4975})
```

<div class="mermaid">
pie title 切分结果
    "A": 5025
    "B": 4975
</div>

可以观察到切分结果还算均衡。

