---
layout: post
title: 使用代码请求网页中的文件上传接口
date: 2021-12-22
tag: Web
---

对于网页中`Content-Type: multipart/form-data`的文件上传接口，使用python可以如下这样请求：

```python
import requests


url = 'example.com'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36'
}
file = 'example.xlsx'
requests.post(url, headers=headers, files={'file': (file, open(file, 'rb'))})
```

