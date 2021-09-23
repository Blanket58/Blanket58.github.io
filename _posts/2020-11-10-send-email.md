---
layout: post
title: 使用代码发送邮件
date: 2020-11-10
tag: [Python, R]
---

## Python

下面为使用Python发送电子邮件的代码示例。

```python
from smtplib import SMTP_SSL
# from smtplib import SMTP
from email.header import Header
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart


def send_email():
    # 创建一个带附件的邮件消息对象
    message = MIMEMultipart()

    # 创建文本内容
    content = '附件中有本月数据请查收'
    text_content = MIMEText(content, 'plain', 'utf-8')
    message['Subject'] = Header('本月数据', 'utf-8')
    # 将文本内容添加到邮件消息对象中
    message.attach(text_content)

    # 读取文件并将文件作为附件添加到邮件消息对象中
    with open('汇总数据.xlsx', 'rb') as f:
        file = MIMEApplication(f.read())
        file.add_header("Content-Disposition", "attachment", filename=("gbk", "", "汇总数据.xlsx"))
        message.attach(file)

    # 创建SMTP对象
    smtper = SMTP_SSL('smtp.163.com')
    # 如果使用世纪互联运营的outlook邮箱，它使用的是SMTP TLS加密
    # smtper = SMTP('smtp.partner.outlook.cn', 587)
    # smtper.starttls()
    sender = 'example@163.com'  # 发件人
    receivers = ['example@qq.com', 'example@icloud.com']  # 收件人
    cc = ['example@qq.com', 'example@icloud.com']  # 抄送人
    bcc = ['example@qq.com', 'example@icloud.com']  # 密送人
    message['From'] = sender
    message['To'] = '; '.join(receivers)
    message['Cc'] = '; '.join(cc)
    message['Bcc'] = '; '.join(bcc)
    smtper.login(sender, 'YOUR PASSWORD')
    smtper.sendmail(sender, receivers + cc + bcc, message.as_string())
    smtper.quit()
    print('发送成功！')
```

## R

下面为使用R发送电子邮件的代码示例。

```R
library(R6)

Emailer <- R6Class('Emailer',
  public = list(
    sender = NULL,
    passwd = NULL,
    receivers = NULL,

    initialize = function(sender, passwd, receivers) {
      suppressMessages(require(emayili))
      suppressMessages(require(magrittr))
      stopifnot(is.character(sender), is.character(passwd), is.character(receivers))

      self$sender <- sender
      self$passwd <- passwd
      self$receivers <- receivers
    },
    send = function(subject_, attachment_) {
      email <- envelope() %>%
        from(self$sender) %>%
        to(self$receivers) %>%
        subject(subject_) %>%
        attachment(attachment_)
      smtp <- server(
        host = 'smtp.163.com',
        username = self$sender,
        password = self$passwd
      )
      smtp(email)
    }
  ))


# NOT RUN
# emailer <- Emailer$new('example@163.com', 'YOUR PASSWORD', 'example@example.com')
# emailer$send('test', 'example.jpg')
# END NOT RUN
```

R本身自带三种面向对象系统，分别是：

- `S3`
- `S4`
- `Reference Class`

上面使用了另一套面向对象系统`R6`，有时间写一下详细的介绍。
