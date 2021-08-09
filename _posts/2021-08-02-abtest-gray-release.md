---
layout: post
title: "ABTest灰度上线"
date: 2021-08-02
description: "ABTest灰度上线"
tag: 数据分析
mermaid: true
---

我们都知道ABTest主要是围绕用户进行的实验，从统计意义上观察用户对不同的产品设计、交互体验、业务流程的反馈，从而指导产品的改进方向。那么很重要的一点就是如何进行分桶。与已知样本总量去切分比例不同，灰度上线时我们不知道用户总量会是多少，因此在线上如何实现具体的切分算法十分关键。

## 一种错误做法

通过UserID的尾数进行划分。这种设计只能进行单层实验，如果考虑长期交叉、连续的实验，这样做有很大的问题。

> 如果在同一分桶中同时进行了X和Y两个实验，那么实验结果就会相互干涉，结果变得不好解释。

## 一种正确做法

目前业界使用最多的就是可重叠分层分桶方法。将流量分成可重叠的多个层，保证流量在不同层中是正交的，即一个用户在每个层中应该分到哪个桶中是独立不相关的。具体来说，如果是二分桶的实验，上一层在一个桶中的用户，理论上应该均匀分布在下一层的两个桶里。这样一份流量通过N个层可以同时进行N个实验，而且实验之间互不干扰，可以提升流量使用率。

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

如此的伪随机分配算法，可以实现实验的复现，同一个用户，无论多少次进入同一个实验，都被会分配进同一个组别。同时，通过给用户名前添加前缀，可以保证不同实验在切分时，同一个用户不会都被分进测试组中，造成分桶有偏的问题。

### Python实现

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

### Java实现

```java
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;


public class App {

    public static boolean gray_release(String prefix, String key, double low, double high) {
        String input = prefix + key;
        double result = 0;
        MessageDigest md = null;
        try {
            md = MessageDigest.getInstance("SHA-256");
            byte[] messageDigest = md.digest(input.getBytes());
            byte[] firstUnsignedLong = new byte[8];
            for (int i = 0; i < 8; i++) {
                firstUnsignedLong[7 - i] = messageDigest[i];
            }
            BigInteger number = new BigInteger(1, firstUnsignedLong);
            BigInteger division = new BigInteger("18446744073709551616", 10);
            result = number.doubleValue() / division.doubleValue();
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return result >= low && result < high;
    }

    public static void main(String[] args) {
        System.out.println(gray_release("test", "326357626", 0, .5));
    }
}
```