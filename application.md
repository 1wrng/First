application

vscode运行程序时： cd+空格+首字母+tab

运行python：首字母 按tab

cmd：win+r 

​            code空格.+回车

pi@10.3.141.1

密码：raspberry

scp  +文件名+pi^^:~/team1/

go_@forward()0~100

ssh +pi^^^

进入team1 python 文件名 回车

vi+文件名 打开

15

46

加仓库：在文件夹中打开git bash

git clone+仓库地址

把要加的东西丢进仓库名文件夹

werlb密码：12345678

在git中进入仓库：git remote add 仓库名 仓库地址

cd /d d：加到d盘

快速上传：目录下webrel_cli.py+上传文件名+—p+密码 ip地址：上传文件名

最好在上传文件的目录里打开cmd

目录：.\

ipconfig查ip地址 无空格

打的是你要上传的ip 第四行

第二行为wifi给你电脑的ip

prt sc然后直接粘贴

打开下端：control+shift+~

加＃：control+/



import network
sta_if = network.WLAN(network.STA_IF)
sta_if.active(True)
sta_if.connect('rjxy', '7uuh8baa')
sta_if.isconnected()
sta_if.ifconfig()

执行完此行命令记得在webrel上把ip改为第一行的

要在arduino上执行命令不能while

快捷上传：python webrepl-master\webrepl_cli.py .\chengxu\main.py -p 12345678 192.168.1.41:main.py

在cmd中一开始直接打不用接斜杠

这里不能在浏览器上连接

进入D盘：直接D：

```
git init                           // 创建Git仓库
git add file                     // 添加文件到git仓库中
git commit -m "note"      // 将添加或修改的文件提交到Git仓库中
git status                       // 查看Git仓库的状态
git diff file                      // 查看文件的修改信息
git log                            // 查看Git仓库中版本的提交日志
git log --pretty=oneline   // 查看Git仓库中版本的提交日志(简略写法)
git reset --hard HEAD^   // 将文件回退到当前版本的前一个版本
```

[![复制代码](http://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

git push -u origin master -f 强制push 

git add 文件夹名/

git add  星号.星号

mkdir 文件名：创文件夹

![1563007771485](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\1563007771485.png)

将库直接clone下来在里面复制黏贴再一条龙就好

jupyter：shift+enter快速运行



data.loc创行

data创列

记得看assignment

dataframe=dataframe(index（行）或columns（列）={'更换目标'：‘更换成的名称})

.descirbe可获取列表中的平均值中位数等等

更换列中数据则：data['更换列名称']



变量名.append(添加变量名称)：向第一个变量名中添加变量

将变量append数据，再将最终所需变量append此变量才能变为二维，直接将最终变量append二维会为一维

用np.array()要在与第一列对齐，要将数据存储起来才能使其变为数组

np.reshape（行，列）从一维变为二维

np.reshape(-1) 变回一维  改变维度,但只是临时的，要使其持续给变量赋值

数组可直接用索引方法获取数据

e.g：print（np.sum(features[:,0])将features数组第一列加起来

切片操作：[,:0] 则此种情况取到的数据就是第0列

knn通过空间位置算距离最近邻得出目标标签

DSTree 通过算熵得出分支模型得出标签

都是先创建模型，然后再通过库来测试



ipynb文件与数据exel表格在同一文件时路径只需.\exel表格名称即可



numpy库学习：

§[https](https://www.runoob.com/numpy/numpy-tutorial.html)[://www.runoob.com/numpy/numpy-tutorial.html](https://www.runoob.com/numpy/numpy-tutorial.html)

.ndim查看数据维度数量

.shape（行数目，列数目）

.size 行数目*列数目

.dtype 数据类型 数据类型对象是用来描述与数组对应的内存区域如何使用，这依赖如下几个方面：

- 数据的类型（整数，浮点数或者 Python 对象）
- 数据的大小（例如， 整数使用多少个字节存储）
- 数据的字节顺序（小端法或大端法）
- 在结构化类型的情况下，字段的名称、每个字段的数据类型和每个字段所取的内存块的部分
- 如果数据类型是子数组，它的形状和数据类型

字节顺序是通过对数据类型预先设定"<"或">"来决定的。"<"意味着小端法(最小值存储在最小的地址，即低位组放在最前面)。">"意味着大端法(最重要的字节存储在最小的地址，即高位组放在最前面)。

.itemsize 查看元素大小 e.g：float64的数组 使用itemsize时结果为8，float64占用64个bits，每个字节长度为8（恒定），所以64/8，占用8个字节，又或是complex32的数组itemsize为4（32/8）



np.empty(shape, dtype = float, order = 'C')

| shape | 数组形状                                                     |
| ----- | ------------------------------------------------------------ |
| dtype | 数据类型，可选                                               |
| order | 有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。 |

```python
np.arange(start, stop, step, dtype)
```

| `start` | 起始值，默认为`0`                                            |
| ------- | ------------------------------------------------------------ |
| `stop`  | 终止值（不包含）                                             |
| `step`  | 步长，默认为`1`                                              |
| `dtype` | 返回`ndarray`的数据类型，如果没有提供，则会使用输入数据的类型。 |

切片：切片还可以包括省略号 **…**，来使选择元组的长度与数组的维度相同。 如果在行位置使用省略号，它将返回包含行中元素的 ndarray。

import numpy as np
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print (a[...,1])   # 第2列元素
print (a[1,...])   # 第2行元素
print (a[...,1:])  # 第2列及剩下的所有元素，不包括第一列



NumPy 算术函数包含简单的加减乘除: **add()**，**subtract()**，**multiply()** 和 **divide()**。

e.g：np.add(数组1，数组2)



np.median(数组1)  取其中位数

np.mean(数组1)  取其算术平均值

np.average(数组1)  取其加权平均值

e.g：数组[1,2,3,4] 权重[4,3,2,1]

```
加权平均值 = (1*4+2*3+3*2+4*1)/(4+3+2+1)
```

np.average([1,2,3,  4],weights =  [4,3,2,1], returned =  True



音频

audio_path = 'DJ Blyatman - Cyka Blyat.mp3'

y, sr = librosa.load(audio_path, sr=采样率)

改变音频速率：librosa.output.write_wav("新的音频名", 音频的信号序列, sr*倍数)

ipd.Audio('音频名')

sr即每秒采样多少个点

加载音频获取 第一个信号值第二个采样率，a,sr =librosa.load....

一般音频为幅度频谱图

切片：数组[行] [列]

knn库：

```python
    import KNN_euklidean
    print(help(KNN_euklidean))
```

获取目标结果



密匙认证需要依靠密匙，首先创建一对密匙（包括公匙和密匙，并且用公匙加密的数据只能用密匙解密），并把公匙放到需要远程服务器上。这样当登录远程 服务器时，客户端软件就会向服务器发出请求，请求用你的密匙进行认证。服务器收到请求之后，先在你在该服务器的宿主目录下寻找你的公匙，然后检查该公匙是 否是合法，如果合法就用公匙加密一随机数（即所谓的challenge）并发送给客户端软件。客户端软件收到 “challenge”之后就用私匙解密再把它发送给服务器。因为用公匙加密的数据只能用密匙解密，服务器经过比较就可以知道该客户连接的合法性。



SSH 为 [Secure Shell](https://baike.so.com/doc/1803865-1907553.html) 的缩写，由 IETF 的[网络](https://baike.so.com/doc/4123457-4322878.html)工作小组(Network Working Group)所制定;SSH 为建立在应用层和传输层基础上的安全协议。SSH 是目前较可靠，专为[远程登录](https://baike.so.com/doc/5696997-5909702.html)会话和其他网络服务提供安全性的协议。利用 SSH 协议可以有效防止远程管理过程中的信息泄露问题。SSH最初是UNIX系统上的一个程序，后来又迅速扩展到其他操作平台。SSH在正确使用时可弥补网络中的漏洞。SSH客户端适用于多种平台。几乎所有UNIX平台-包括HP-UX、[Linux](https://baike.so.com/doc/5349227-5584683.html)、[AIX](https://baike.so.com/doc/3676093-3863703.html)、[Solaris](https://baike.so.com/doc/6789539-7006148.html)、[Digital](https://baike.so.com/doc/5447842-5686210.html) [UNIX](https://baike.so.com/doc/5410818-5648913.html)、[Irix](https://baike.so.com/doc/7555143-7829236.html)，以及其他平台，都可运行SSH。



$ git rm -r --cached target              # 删除target文件夹
$ git commit -m '删除了target'        # 提交,添加操作说明

记得要push



## Arduino

void *memset(void *s, int ch, [size_t](https://baike.so.com/doc/6847447-24969833.html) n);

函数解释:将s中当前位置后面的n个字节 (typedef unsigned int size_t )用 ch 替换并返回 s 。



# 1. 结构体(struct)

### 1.1 结构体的概念

- **结构体(struct)：**是由一系列具有相同类型或不同类型的数据构成的数据集合，叫做结构。
- **结构体(struct)：**是一种复合数据类型，结构类型。
- **注：“结构”是一种构造类型，它是由若干“成员”组成的。 每一个成员可以是一个基本数据类型或者又是一个构造类型。 结构即是一种“构造”而成的数据类型， 那么在说明和使用之前必须先定义它，也就是构造它。如同在说明和调用函数之前要先定义一样。**



类的大括号后记得加分号