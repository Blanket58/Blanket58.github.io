---
layout: post
title: wsl2的ip转发
date: 2020-11-27
tag: Linux
---

win10中的wsl2与wsl1不同，2是一个纯正的虚拟机，因此如果我们在wsl2中搭建一个web应用，并想从别的设备访问的时候，使用本机的ip地址是访问不到的，因为wsl2中的ip地址与我们本机的ip地址是不一致的。

win10的powershell自带了ip桥接命令，我们首先需要获得wsl2中的ip地址，在wsl2中运行下面的命令：

```bash
ifconfig eth0
```

结果如下图所示，其中红框内的ip为真正可以在本机访问wsl2内web应用的ip：

![2020-11-27-wsl2-ip-bridge-1](/assets/2020-11-27-wsl2-ip-bridge-1.png)

接下来要做的，就是将wsl2的ip地址与端口桥接至本机ip上，下面的代码需要使用管理员权限在powershell中运行。举例说明，如果我们想将在wsl2中运行的mysql服务桥接至本机ip，mysql默认使用3306端口，因此代码为：

```powershell
netsh interface portproxy add v4tov4 listenport=3306 connectaddress=192.168.141.126 connectport=3306 listenaddress=* protocol=tcp
```

当我们不再需要使用wsl2内的web服务时，可以运行下面的命令，删除ip桥接：

```powershell
netsh interface portproxy delete v4tov4 listenport=3306 protocol=tcp
```

需要注意的是，wsl2内的ip地址是动态的，wsl2的每次重启都会导致ip变化，因此我们需要定时执行桥接命令，我在这里用python写了一个脚本：

```python
from subprocess import run, PIPE


ip = run('wsl bash -c "ifconfig eth0 | grep -E -o \'inet 192.168.[0-9]{3}.[0-9]{3}\' | sed \'s/^.*inet //g\'"', stdout=PIPE).stdout.decode().strip()
run('powershell.exe netsh interface portproxy delete v4tov4 listenport=3306 protocol=tcp')
run(f'powershell.exe netsh interface portproxy add v4tov4 listenport=3306 connectaddress={ip} connectport=3306 listenaddress=* protocol=tcp')
print('Done.')
```

注意：python脚本同样需要使用管理员权限运行。
