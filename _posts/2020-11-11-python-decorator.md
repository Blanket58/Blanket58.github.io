---
layout: post
title: "Python装饰器"
date: 2020-11-11
description: "一些Python装饰器"
tag: Python
---

整理了一些平时经常会用到的装饰器。

```python
import logging
import traceback
from email.header import Header
from email.mime.text import MIMEText
from functools import wraps
from smtplib import SMTP
from time import time, localtime, strftime

from decorator import decorator

__all__ = ['retry', 'timer', 'TaskHandler']


def stream_logger_factory(func):
    """Factory to create stream logger."""
    assert callable(func) or isinstance(func, str), 'Only allowed callable object or string.'
    if callable(func):
        name = func.__name__.upper()
    else:
        name = func.upper()
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


@decorator
def retry(func, max_retry=5, logger=None, *args, **kwargs):
    """Not stop retrying until reach max limit."""
    if not logger:
        logger = stream_logger_factory(func)

    error = None
    for i in range(max_retry):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f'Retrying [{i + 1} / {max_retry}]')
            error = e
    raise error


@decorator
def timer(func, logger=None, *args, **kwargs):
    """Calculate how long the function runs."""
    if not logger:
        logger = stream_logger_factory(func)

    start = int(round(time() * 1000))
    logger.info("Start")
    result = func(*args, **kwargs)
    end = int(round(time() * 1000)) - start
    end /= 1000
    m, s = divmod(end, 60)
    h, m = divmod(m, 60)
    logger.info("Done")
    logger.info("Total execution time: %d:%02d:%02d" % (h, m, s))
    return result


class TaskHandler:
    """自动任务处理器

    功能：
    1. 留日志
    2. 自动发送任务执行成功或失败邮件通知（附带报错信息）
    3. 任务计时
    """

    def __init__(self, name, sender, passwd, receivers, send_success_email=True):
        """
        :param name: str 任务名
        :param sender: str 发件人
        :param passwd: str 发件人邮箱密码
        :param receivers: list 收件人
        """
        self.sender = sender
        self.receivers = receivers
        self.passwd = passwd
        self.task_name = name
        self.send_success_email = send_success_email
        self.logger = self.__create_logger(name)

    @staticmethod
    def __create_logger(name):
        """
        Create a logging object and return it.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{name}.log', encoding='utf-8')
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def __emailer(self, subject, message):
        message['From'] = self.sender
        message['To'] = '; '.join(self.receivers)
        message['Subject'] = Header(subject, 'utf-8')
        # Outlook中国运营商SMTP TLS邮件服务器地址与端口
        smtper = SMTP('smtp.partner.outlook.cn', 587)
        smtper.starttls()
        smtper.login(self.sender, self.passwd)
        smtper.sendmail(self.sender, self.receivers, message.as_string())
        smtper.quit()

    def __send_exception(self, content):
        subject = f'Failed -> {self.task_name}'
        message = MIMEText(content, 'plain', 'utf-8')
        self.__emailer(subject, message)

    def __send_success(self, content):
        subject = f'Success -> {self.task_name}'
        message = MIMEText(f'Task {self.task_name} success.\n{content}', 'plain', 'utf-8')
        self.__emailer(subject, message)

    def handler(self, func):
        """
        A decorator that wraps the passed in function and logs
        exceptions if one occur.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = int(round(time() * 1000))
            try:
                self.logger.info('-' * 30)
                self.logger.info('***Start***')
                func(*args, **kwargs)
                if self.send_success_email:
                    self.__send_success(f'Complete at {strftime("%Y-%m-%d %H:%M:%S", localtime())}.')
            except Exception as e:
                self.logger.exception(e)
                self.__send_exception(str(e) + '\n' + traceback.format_exc())
                raise e
            finally:
                self.logger.info('***Exit***')
                end_ = int(round(time() * 1000)) - start
                end_ /= 1000
                m, s = divmod(end_, 60)
                h, m = divmod(m, 60)
                self.logger.info('Total execution time: %d:%02d:%02d' % (h, m, s))

        return wrapper
```

其中`TaskHandler`的使用方法如下：

```python
th = TaskHandler(name, sender, passwd, receivers)
logger = th.logger


@th.handler
def main():
    logger.info('Test')


if __name__ == '__main__':
    main()
```

运行完毕后，会自动发送邮件给收件人通知任务运行成功，同时本地还会生成一个`Test.log`文件，内容如下：

```
2020-11-11 10:18:02,240 - Test - INFO - ------------------------------
2020-11-11 10:18:02,240 - Test - INFO - ***Start***
2020-11-11 10:18:02,240 - Test - INFO - Test
2020-11-11 10:18:45,643 - Test - INFO - ***Exit***
2020-11-11 10:18:45,643 - Test - INFO - Total execution time: 0:00:43
```
