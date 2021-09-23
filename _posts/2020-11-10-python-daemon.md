---
layout: post
title: 使用python实现一个守护进程
date: 2020-11-10
tag: Python
---

如何实现自动重启一个运行失败的Python脚本呢？下面的代码实现了这个功能，如果任务运行失败，会间隔30秒后自动重启，最多执行三次。

```python
from argparse import ArgumentParser
from time import sleep
from subprocess import run
from os import listdir

parser = ArgumentParser(prog='Python daemon', description='Restart python script if failed after 30 seconds, retry 3 times.')
parser.add_argument('file', type=str, help='The python script file name.')
args = parser.parse_args()

file = args.file
if file not in listdir('.'):
    raise FileNotFoundError

count = 0
while count < 3:
    count += 1
    exit_code = run(['python', file])
    if exit_code.returncode == 0:
        break
    else:
        print('Retry after 30 seconds.')
        sleep(30)
```

将上面的代码保存为文件`daemon.py`。以windows为例，遇到想要运行失败自动重启的Python任务脚本，只需将下面的代码保存为一个`.bat`文件，并配置任务计划程序即可。

```python
python daemon.py your_script_file.py
```
