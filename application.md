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



人工智能学习：天池AI



密匙认证需要依靠密匙，首先创建一对密匙（包括公匙和密匙，并且用公匙加密的数据只能用密匙解密），并把公匙放到需要远程服务器上。这样当登录远程 服务器时，客户端软件就会向服务器发出请求，请求用你的密匙进行认证。服务器收到请求之后，先在你在该服务器的宿主目录下寻找你的公匙，然后检查该公匙是 否是合法，如果合法就用公匙加密一随机数（即所谓的challenge）并发送给客户端软件。客户端软件收到 “challenge”之后就用私匙解密再把它发送给服务器。因为用公匙加密的数据只能用密匙解密，服务器经过比较就可以知道该客户连接的合法性。**(如果只用密码登陆就容易被暴力破解)**



SSH 为 [Secure Shell](https://baike.so.com/doc/1803865-1907553.html) 的缩写，由 IETF 的[网络](https://baike.so.com/doc/4123457-4322878.html)工作小组(Network Working Group)所制定;SSH 为建立在应用层和传输层基础上的安全协议。SSH 是目前较可靠，专为[远程登录](https://baike.so.com/doc/5696997-5909702.html)会话和其他网络服务提供安全性的协议。利用 SSH 协议可以有效防止远程管理过程中的信息泄露问题。SSH最初是UNIX系统上的一个程序，后来又迅速扩展到其他操作平台。SSH在正确使用时可弥补网络中的漏洞。SSH客户端适用于多种平台。几乎所有UNIX平台-包括HP-UX、[Linux](https://baike.so.com/doc/5349227-5584683.html)、[AIX](https://baike.so.com/doc/3676093-3863703.html)、[Solaris](https://baike.so.com/doc/6789539-7006148.html)、[Digital](https://baike.so.com/doc/5447842-5686210.html) [UNIX](https://baike.so.com/doc/5410818-5648913.html)、[Irix](https://baike.so.com/doc/7555143-7829236.html)，以及其他平台，都可运行SSH。



$ git rm -r --cached target              # 删除target文件夹
$ git commit -m '删除了target'        # 提交,添加操作说明

记得要push

**强人工智能和弱人工智能**

[强人工智能](https://baike.so.com/doc/6296321-6509841.html)观点认为有可能制造出真正能推理(Reasoning)和解决问题(Problem_solving)的智能机器，并且，这样的机器能将被认为是有知觉的，有自我意识的。可以独立思考问题并制定解决问题的最优方案，有自己的价值观和世界观体系。

弱人工智能是指不能制造出真正地推理(Reasoning)和解决问题(Problem_solving)的智能机器，这些机器只不过看起来像是智能的，但是并不真正拥有智能，也不会有自主意识。



**感知机**

感知机由Rosenblatt在1957年提出,是一种二类线性分类模型。输入一个实数值的n维向量（特征向量），经过线性组合，如果结果大于某个数，则输出1，否则输出-1.



**人工神经网络**

人工神经网络（Artificial Neural Networks，简写为ANNs）是一种模仿动物神经网络行为[特征](https://baike.so.com/doc/152515.html)，进行分布式并行信息处理的算法[数学模型](https://baike.so.com/doc/5513175.html)。这种网络依靠系统的复杂程度，通过调整内部大量节点之间相互连接的关系，从而达到处理信息的目的，并具有自学习和自适应的能力。



**深度学习**

通过组合底层特征形成更加抽象的高层并表示属性类型或特征，以发现数据的分布式特征表示；还能利用空间相对关系，减少参数数目以提高训练性能，其动机在于建立模拟人脑进行分析学习的神经网络



**强化学习**(reinforcement learning)，又称再励学习、评价学习，是一种重要的[机器](https://baike.so.com/doc/6002736-6215713.html)学习方法，在智能控制机器人及分析预测等领域有许多应用。在机器学习分类中共有监督学习、无监督学习和强化学习三种类型。



智能体先对环境进行观察，然后对环境做出动作，然后会得到环境的奖赏和再一次观察，不断重复，动作具有长期性质，有时暂时做一个动作会牺牲一些小利益，但是结合多个动作，利益会得到最大化







## Arduino

void *memset(void *s, int ch, [size_t](https://baike.so.com/doc/6847447-24969833.html) n);

函数解释:将s中当前位置后面的n个字节 (typedef unsigned int size_t )用 ch 替换并返回 s 。



# 1. 结构体(struct)

### 1.1 结构体的概念

- **结构体(struct)：**是由一系列具有相同类型或不同类型的数据构成的数据集合，叫做结构。
- **结构体(struct)：**是一种复合数据类型，结构类型。
- **注：“结构”是一种构造类型，它是由若干“成员”组成的。 每一个成员可以是一个基本数据类型或者又是一个构造类型。 结构即是一种“构造”而成的数据类型， 那么在说明和使用之前必须先定义它，也就是构造它。如同在说明和调用函数之前要先定义一样。**





struct不能定义函数，而class可以



类的大括号后记得加分号

构造函数名称与类一致，析构函数在局部对象被删除前（即构造函数执行后对象被删除）前执行

析构函数即构造函数前加~

在类中建立void函数

可在setup中建立类的对象来调用类中的函数，e.g：类名 对象名（建立对象）

​											对象名.函数名（调用函数）

也可建立全局对象

类中：

Led（int userLedPin）



构造：

Led::Led(int userLedPin){

ledPin=userLedPin

pinMode(ledpin,OUTPUT)

}

Setup中：

Led myled2(7)      此时7被赋值到userLedPin中再被赋值到ledPin



同时，Led::Led(int userLedPin): ledpin(userLedpin)就是此构造函数中的userLedPin被ledPin赋值



可有多个构造函数

类的封装：通过在私有成员中设立变量，再通过构造函数在外获取私有变量

e.g:         private:

​			ledpin=2



**int** Led::getLedPin(){

​	return ledpin

}               此时getLedPin（）就被ledpin赋值

void Led::setLedPin(int userLedpin){

​	ledPin=userLedpin

​	pinMode(ledPin, OUTPUT)

}     		 此时可在后面的setup中使用此函数改变默认引脚；     这两个函数即是类的封装，不给这两段函数用户即无法获取私有成员中的变量。 在setup中先用setledpin再用getledpin，setledpin可对私有成员变量直接修改。



建立库：

h文件为头文件，cpp为源文件；此项目中h文件中为类，cpp文件中为对类的函数的诠释

建立两个文档，一个为cpp，一个为h，两个文件的前缀都为类的名称



h：#ifndef _文件名称（全大写)  _H _  (无空格)

​      #define  与上面一致

#include<Arduino.h>



cpp: #include "h文件全名“

而总文件也是#include “h文件全名”



创建对象时是用类来创建对象，不是构造函数

类的继承

只有公共成员能被继承，私有不能

在cpp文件中 输入：class PwmLed （子类名称）: 子类成员类型（大多数都为public） Led（父类名称）{}；

在cpp文件中操作完后直接到总文件里即可用子类

在诠释类文件时要引用类中的函数（包括子类引用父类函数）都不需要对象直接引用



在子类中建立了一个与父类成员函数重名时，子类函数会覆盖掉父类的成员函数，称为函数重载。



想添加库需要向libraries中建立库的名称的文件夹包含cpp文件和h文件即可

想在库中加载实例程序需要在库文件夹中加一个examples文件夹，再向其中加一个和库名字一样的文件夹，在其中加一个与库名字一样的ino文件（程序）。

当cpp和h文件添加到库文件夹中，可用<>

而与程序放在一起时，便用" "，而被添加到库文件后，" "也适用。




同时在库文件中建立txt文件 通过类建立对象KEYWORD1（橙色加粗）

通过成员函数建立对象KEYWORD2（橙色不加粗）



总结：类为具有相同属性和行为的对象统称，对象为具体化表现形式

封装是为了把对象的设计者和对象的使用者分开。



class Car{

​	public:

​		car()

​	};

Car::car{

}



## Tensorflow2.0学习

在tf2.0中，所有变量或constant都存在于动态，不需要run系统自动执run操作，也就是可以直接print

用

```python
tf.__version__
```

可查看当前tensorflow版本



#### 线性回归

```python
x=data.^
y=data.^
model=tf.keras.Sequential()
model.add(tf.keras.Dense(1,input_shape=(1,)))#此时为一维空间模型，前一个1代表输出数据的维度，后一个1表示输入数据的维度，1后面有个逗号表示这是元组

model.summary()#显示模型，param中的数据表示有几个参数
model.compile(optimizer='adam'
             loss='mse')#compile是自定义loss函数，optimizer是优化器，mse是均方差
history=model.fit(x,y,epoches=100)#100为训练次数，此过程loss会不断减小
model.predict(x)
model.predict(pd.Series([20]))#此处即当x=20时预测y的结果
```

list和ndarray和Tensor的区别 	

![image-20200312235735591](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200312235735591.png)

多层感知机（神经网络第一次实现）

激活函数即看输入值是否达到阈值（拟合非线性问题）

![image-20200331230524617](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200331230524617.png)

![image-20200331230606841](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200331230606841.png)

![image-20200331230632367](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200331230632367.png)

![image-20200331230650891](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200331230650891.png)

```python
data=pd.read_csv('.\Advertising.csv')
data.head()
x=data.iloc[:,1：-1]#第二行到第四行
y=data.iloc[:,-1]#第五行
model=tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=(3,),activation='relu'),
      tf.keras.layers.Dense(1)                   ])#Sequential是顺序模型，直接在里面写dense可省去add。1个隐藏层中有10个单元，input_shape=（3，）是指输入的数据为3个特征,后一个dense是1指输出的1个标签
model.summary()
model.compile(optimizer='adam',
              loss='mse'
)

model.fit(x,y,epochs=100)
test=data.iloc[:10,1:-1]
model.predict(test)#预测x的值


```

逻辑回归与交叉熵

sigmoid是概率分布激活函数

![image-20200401003132761](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401003132761.png)

mse用于惩罚与损失在同一数量级时，而对于分类问题用交叉熵处理loss更好

![image-20200317004237305](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200317004237305.png)

![image-20200316080343483](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200316080343483.png)

二分类问题

```python
data=pd.read_csv('.\credit-a.csv',header=none)#表示默认加个第一行分别为1234567作为列表序列，即把原来表格第一行往下移
data.head()
data.iloc[:,-1].value_counts()#查看最后一列的数据分布情况
x=data.iloc[:,:-1]#最后一列的前面都作为特征
y=data.iloc[:,-1].replace(-1,0)#将最后一列中的-1都换成0
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4,input_shape=(15,),activation='relu'))
model.add(tf.keras.layers.Dense(4,activation='relu'))#从第二层开始便不需要说明特征数
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))#sigmoid为概率分布激活函数，此层为输出层
model.summary()#有两层隐藏层，一层输出层
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])#loss使用二元交叉熵算法，metrics是在计算准确率
history = model.fit(x,y,epochs=100)
history.history.keys()#调用字典，记录哪两个数据在变化
plt.plot(history.epochs,history.history.get('loss'))#调用字典中的loss数据

```



Softmax层(sigmoid对单个样本，softmax覆盖所有样本)

![image-20200331192133490](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200331192133490.png)

![image-20200322152554273](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200322152554273.png)

![image-20200322180054564](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200322180054564.png)

softmax每个样本的分量和为1

Fashion MNIST可以直接在tensorflow中引入

```python
(train_image,train_lable),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()#与上面一句是同一行
print(test_image.shape)
print(train_image.shape)#60000张图像，28x28的图像
plt.imshow(train_image[0])
np.max(train_image[0])
print(train_lable)#用数字对lable中的图像顺序标类
train_image=train_image/255
test_image=test_image/255 #对test_image映射，使其取值范围变为0到1
model=tf.keras.Sequential()
model.add((tf.keras.layers.Flatten(input_shape=(28,28))) #变为28x28的向量
model.add(tf.keras.layers.Dense(128,activation='relu'))#输出128个隐藏单元，此层为隐藏层。
model.add(tf.keras.layers.Dense(10,activation='softmax'))#输出十个数值变为概率分布，10个概率和为1，此层为输出层
model.compile(optimizer=‘adam'，
              loss='sparse_catogorical_crossentropy'，
              metrics=['acc'])#lable使用数字编码，用sparse_catogorical_crossentropy
          
model.fit(train_image,train_lable, epochs=5)
model.evaluate(test_image,test_lable)
```

迁移学习

![image-20200401215418024](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401215418024.png)

![image-20200401215539235](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401215539235.png)

![image-20200401215718761](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401215718761.png)

![image-20200401215753647](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401215753647.png)

![image-20200401220047322](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401220047322.png)

![image-20200401220241290](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401220241290.png)





独热编码（准确率和loss等几乎一样，只是编码方式有差别）

```python
train_lable_onehot=tf.keras.utils.to_categorical(train_lable)
```



优化函数、学习速率与反向传播算法

![image-20200401230954610](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401230954610.png)

![image-20200401231221437](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401231221437.png)

![image-20200401231247189](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401231247189.png)

![image-20200401231710191](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401231710191.png)

![image-20200401231915939](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401231915939.png)

![image-20200401231959914](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401231959914.png)

![image-20200401232020531](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401232020531.png)

![image-20200401232119802](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401232119802.png)

![image-20200401232136394](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401232136394.png)

![image-20200401232250542](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401232250542.png)

![image-20200401232318581](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401232318581.png)

![image-20200401232416060](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401232416060.png)

```python
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['acc'])#这里使adam默认用的lr参数从0.001变为0.01
```



网络优化与超参数选择

网络容量可认为与网络中的可训练参数成正比

![image-20200401233258811](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401233258811.png)

![image-20200401233546116](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401233546116.png)

![image-20200401233614673](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401233614673.png)

![image-20200401233652889](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401233652889.png)

![image-20200401233741941](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200401233741941.png)

params是可训练参数

***增加网络拟合能力可以提升模型的准确率***

**过拟合**：在train数据随着epochs增加loss一直递减，但是test到中间开始loss有上升，这种情况叫做过拟合，同时过拟合时会使train的acc明显大于test的acc

**欠拟合**：训练数据得分较低，而测试数据得分相对更低



理想模型在过拟合和欠拟合的界限之间



![image-20200402023515643](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200402023515643.png)

![image-20200402023804257](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200402023804257.png)

![image-20200402023912813](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200402023912813.png)

![image-20200402024042126](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200402024042126.png)

抑制过拟合的最好方法是增加训练数据，使用dropout是在没有训练数据的情况下

![image-20200402024526295](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200402024526295.png)



RNN神经循环网络

![image-20200406122000514](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406122000514.png)

![image-20200406122023759](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406122023759.png)

![](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406142716025.png)



LSTM

![image-20200406164409936](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406164409936.png)

![image-20200406164421360](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406164421360.png)

![image-20200406164428264](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406164428264.png)

![image-20200406164433823](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406164433823.png)

![image-20200406164439965](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406164439965.png)

![image-20200406164445534](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406164445534.png)

![image-20200406164451733](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200406164451733.png)

LSTM引入选择性遗忘，对前一状态通过遗忘门结合当今状态通过点积和加法得到输出结果



批标准化

![image-20200407000418529](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000418529.png)

![image-20200407000426063](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000426063.png)

![image-20200407000431728](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000431728.png)

![image-20200407000438147](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000438147.png)

![image-20200407000442598](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000442598.png)

![image-20200407000446926](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000446926.png)

![image-20200407000451384](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000451384.png)

![image-20200407000455660](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000455660.png)

![image-20200407000500164](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000500164.png)

![image-20200407000508622](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407000508622.png)

![image-20200407221736521](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407221736521.png)

![image-20200407221742727](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407221742727.png)

![image-20200407221747629](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407221747629.png)

![image-20200407221754734](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407221754734.png)

![image-20200407221819202](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200407221819202.png)

减均值除方差

目前效果较好的lstm1

```python
model=keras.models.Sequential([
    keras.layers.Embedding(10000,50,input_length=300),
#     keras.layers.GlobalAveragePooling1D(),
    keras.layers.Bidirectional(keras.layers.LSTM(units=64,return_sequences= True)),
    keras.layers.Bidirectional(keras.layers.LSTM(units=64,return_sequences= False)),#return_sequences,True是采用所有输出，False是采用最后一步输出
    keras.layers.Dense(64,activation='relu'),
#     keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64,activation='relu'),
#     keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid'),
#     keras.layers.BatchNormalization(),
    
])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])
 history=model.fit(x_train,y_train,epochs=15,batch_size=64,validation_split=0.3)
```



目前效果较好的lstm2

```python
model=keras.models.Sequential([
    keras.layers.Embedding(10000,50,input_length=300),
#     keras.layers.GlobalAveragePooling1D(),
#     keras.layers.Bidirectional(keras.layers.LSTM(units=64,return_sequences= True)),
    keras.layers.Bidirectional(keras.layers.LSTM(units=64,return_sequences= False)),#return_sequences,True是采用所有输出，False是采用最后一步输出
#     keras.layers.BatchNormalization(),
    keras.layers.Dense(64,activation='relu'),

    keras.layers.Dropout(0.3),
#     keras.layers.BatchNormalization(),
    keras.layers.Dense(64,activation='relu'),
    
    keras.layers.Dropout(0.3),
#     keras.layers.BatchNormalization(),
    keras.layers.Dense(32,activation='relu'),

    keras.layers.Dropout(0.3),
#     keras.layers.BatchNormalization(),
    keras.layers.Dense(1,activation='sigmoid'),

    
])
model.summary()
```

目前比赛样本效果还行的lstm

```python
model=keras.models.Sequential([
    keras.layers.Embedding(10000,50,input_length=300),
#     keras.layers.GlobalAveragePooling1D(),
    keras.layers.Bidirectional(keras.layers.LSTM(units=128,return_sequences= True)),
    keras.layers.Bidirectional(keras.layers.LSTM(units=128,return_sequences= False)),#return_sequences,True是采用所有输出，False是采用最后一步输出
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64,activation='relu'),

    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32,activation='relu'),

    
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),   
    keras.layers.Dense(32,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),

    
    
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1,activation='sigmoid',kernel_regularizer=keras.regularizers.l2(0.001)),

    
])
model.summary()
```

