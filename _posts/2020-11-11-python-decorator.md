---
layout: post
title: "使用python装饰器处理错误"
date: 2020-11-11
description: "使用python装饰器实现自动化任务日志持久化，并自动触发错误报警邮件。"
tag: Python
---

下面是如何使用Python装饰器实现自动化任务日志持久化，并自动触发错误报警邮件。

```python
import logging
import traceback
from email.header import Header
from email.mime.text import MIMEText
from smtplib import SMTP
from time import time, localtime, strftime

__all__ = ['ExceptionLogger']


class ExceptionLogger:

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
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    def __send_exception_email(self, content):
        subject = f'Failed -> {self.task_name}'
        message = MIMEText(content, 'plain', 'utf-8')
        self.__emailer(subject, message)

    def __send_success_email(self, info):
        subject = f'Success -> {self.task_name}'
        message = MIMEText(f'Task {self.task_name} success.\n{info}', 'plain', 'utf-8')
        self.__emailer(subject, message)

    def handler(self, func):
        """
        A decorator that wraps the passed in function and logs
        exceptions if one occur.
        """

        def wrapper(*args, **kwargs):
            start = int(round(time() * 1000))
            try:
                self.logger.info('-' * 30)
                self.logger.info('***Start***')
                func(*args, **kwargs)
                if self.send_success_email:
                    self.__send_success_email(f'Complete at {strftime("%Y-%m-%d %H:%M:%S", localtime())}.')
            except Exception as e:
                self.logger.exception(e)
                self.__send_exception_email(str(e) + '\n' + traceback.format_exc())
                raise RuntimeError
            finally:
                self.logger.info('***Exit***')
                end_ = int(round(time() * 1000)) - start
                end_ /= 1000
                m, s = divmod(end_, 60)
                h, m = divmod(m, 60)
                self.logger.info('Total execution time: %d:%02d:%02d' % (h, m, s))

        return wrapper
```

将上面的代码保存为文件`decor.py`，在使用时，只需将其套在需要监控的函数上即可，如下：

```python
from decor import ExceptionLogger

el = ExceptionLogger(name, sender, passwd, receivers)
logger = el.logger


@ el.handler
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
