---
layout: post
title: 使用PySimpleGUI编写多线程桌面应用
date: 2021-02-25
tag: Python
---

下面是一个使用PySimpleGUI编写桌面应用的小例子，底层是python自带的tkinter，代码是可以跨平台的。

这是这个项目的文件树：

```text
├─utils/
|   ├─__init__.py
|   ├─db.py
|   └─tools.py
├─app.py
└─code.sql
```

其中`code.sql`是我们的取数SQL模板，模板参数使用jinja2进行渲染；`db.py`文件，是[另外一篇文章中写的数据库连接类](https://blanket58.github.io/2020/12/impyla/)；`utils`文件夹下的另一个文件`tools.py`内容如下：

```python
import base64
from logging import StreamHandler


class GuiStreamHandler(StreamHandler):

    def emit(self, record):
        print(self.format(record))


def file2base64(file):
    with open(file, 'rb') as f:
        img = f.read()
    return base64.b64encode(img)


def check_string(s):
    reserved_words = (
        'add', 'aggregate', 'all', 'alter', 'analytic', 'and', 'anti',
        'api_version', 'array', 'as', 'asc', 'avro', 'between', 'bigint',
        'binary', 'boolean', 'by', 'cached', 'case', 'cast', 'change', 'char',
        'class', 'close_fn', 'column', 'columns', 'comment', 'compute',
        'create', 'cross', 'current', 'data', 'database', 'databases', 'date',
        'datetime', 'decimal', 'delimited', 'desc', 'describe', 'distinct',
        'div', 'double', 'drop', 'else', 'end', 'escaped', 'exists', 'explain',
        'external', 'false', 'fields', 'fileformat', 'finalize_fn', 'first',
        'float', 'following', 'for', 'format', 'formatted', 'from', 'full',
        'function', 'functions', 'grant', 'group', 'having', 'if', 'in',
        'init_fn', 'inner', 'inpath', 'insert', 'int', 'integer',
        'intermediate', 'interval', 'into', 'invalidate', 'is', 'join', 'last',
        'left', 'like', 'limit', 'lines', 'load', 'location', 'map',
        'merge_fn', 'metadata', 'not', 'null', 'nulls', 'offset', 'on', 'or',
        'order', 'outer', 'over', 'overwrite', 'parquet', 'parquetfile',
        'partition', 'partitioned', 'partitions', 'preceding', 'prepare_fn',
        'produced', 'range', 'rcfile', 'real', 'refresh', 'regexp', 'rename',
        'replace', 'returns', 'revoke', 'right', 'rlike', 'role', 'roles',
        'row', 'rows', 'schema', 'schemas', 'select', 'semi', 'sequencefile',
        'serdeproperties', 'serialize_fn', 'set', 'show', 'smallint', 'stats',
        'stored', 'straight_join', 'string', 'struct', 'symbol', 'table',
        'tables', 'tblproperties', 'terminated', 'textfile', 'then',
        'timestamp', 'tinyint', 'to', 'true', 'unbounded', 'uncached', 'union',
        'update_fn', 'use', 'using', 'values', 'varchar', 'view', 'when',
        'where', 'with')
    assert s.isalnum() or s == '', '仅可输入字母与数字的组合'
    s = s.lower()
    for x in reserved_words:
        assert x not in s, '非法字符串'
```

这个文件提供以下三个功能：

- 将logging输出重定向至stdout的`GuiStreamHandler`类
- 将png文件编码为base64的函数`file2base64`，我们的GUI应用如果需要图标，就需要将图标编码为base64
- 检查SQL的查询条件输入没有非法字符串，防止SQL注入的函数`check_string`

下面就是主文件`app.py`:

```python
import logging
from getpass import getuser
from os.path import abspath
from threading import Thread

import PySimpleGUI as sg
from xlwings import App
from jinja2 import Template

from utils.db import Impala
from utils.tools import GuiStreamHandler, check_string


logger = logging.getLogger('MAIN')
logger.setLevel(logging.DEBUG)
fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(fmt)
gsh = GuiStreamHandler()
gsh.setFormatter(formatter)
logger.addHandler(gsh)


def operation_thread(sql, window):
    logger.info('Job start at background.')
    db = Impala()

    app = App(visible=False)
    wb = app.books.active
    wb.sheets.add('xxx')
    wb.sheets.add('yyy')
    wb.sheets['Sheet1'].delete()

    df = db.fetch(sql, tag='1')
    wb.sheets['xxx'].range('A1').options(index=False).value = df
    wb.sheets['xxx'].range((1, 1), (1, df.shape[1])).api.Interior.ColorIndex = 8
    wb.sheets['xxx'].range('D:D').number_format = '0'
    wb.sheets['xxx'].autofit('c')
    wb.sheets['xxx'].api.Columns.AutoFilter(1)
    logger.info('xxx done.')

    df = db.fetch(sql, tag='2')
    wb.sheets['yyy'].range('A1').options(index=False).value = df
    wb.sheets['yyy'].range((1, 1), (1, df.shape[1])).api.Interior.ColorIndex = 8
    wb.sheets['yyy'].autofit('c')
    wb.sheets['yyy'].api.Columns.AutoFilter(1)
    logger.info('yyy done.')

    wb.save('RESULT.xlsx')
    wb.close()
    app.quit()

    db.close()
    logger.info('Job finished.')
    logger.info(f'File saved at {abspath("RESULT.xlsx")}')
    window.write_event_value('-JOB-', '')


def main():
    with open('code.sql', 'r', encoding='utf-8') as f:
        sql_templ = Template(f.read())

    sg.theme('SandyBeach')

    layout = [
        [sg.Text('拉取信息')],
        [sg.MLine(size=(80, 15), key='-LOG-', reroute_stdout=True,
                  write_only=True, autoscroll=True, auto_refresh=True)],
        [sg.Text('任选以下其一填写')],
        [
            sg.Text('用户的username'),
            sg.Input(key='-USERNAME-', focus=True, size=(15, 1)),
            sg.Text('或'),
            sg.Text('用户的userid'),
            sg.Input(key='-USERID-', size=(15, 1))
        ],
        [
            sg.Button('Run'),
            sg.Button('Exit')
        ]
    ]

    window = sg.Window('拉取信息', layout, finalize=True)

    if getuser() != 'admin':
        user_confirm = sg.popup_error('计算机未以指定用户名登录，禁止使用', keep_on_top=True, no_titlebar=True)
    else:
        user_confirm = None

    thread = None
    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, 'Exit') or user_confirm == 'Error':
            break
        elif event == 'Run' and not thread:
            try:
                userid = values['-USERID-']
                username = values['-USERNAME-']
                # Check for preventing sql injection
                assert userid == '' or userid.isdigit(), 'userid必须为Integer'
                check_string(username)
                # ---
                assert not all([userid, username]) and any([userid, username]), 'userid和username必须有一个为空'
                thread = Thread(target=operation_thread,
                                args=(sql_templ.render(userid=userid, username=username), window), daemon=True)
                thread.start()
                sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, background_color='white', transparent_color='white',
                                  time_between_frames=100)
            except Exception as e:
                logger.error(e)
        elif event == '-JOB-':
            thread.join()
            sg.popup_animated(None)  # stop animination if one is running
            thread = None
        if thread is not None:
            sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, background_color='white', transparent_color='white',
                              time_between_frames=100)
    window.close()


if __name__ == '__main__':
    main()
```

运行`app.py`文件，就会有如下的窗口弹出：

![2021-02-25-pysimplegui-multithreading](/assets/2021-02-25-pysimplegui-multithreading.png)
