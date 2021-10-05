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



### CNN

![image-20200923232338424](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923232338424.png)

![image-20200923232405130](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923232405130.png)

![image-20200923232424479](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923232424479.png)

![image-20200923232453850](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923232453850.png)

![image-20200923232513354](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923232513354.png)

原理

![image-20200923234036245](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234036245.png)

![image-20200923234045116](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234045116.png)

![image-20200923234052525](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234052525.png)

![image-20200923234059844](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234059844.png)

![image-20200923234112847](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234112847.png)

![image-20200923234118984](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234118984.png)

![image-20200923234125219](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234125219.png)

![image-20200923234131185](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234131185.png)

![image-20200923234137052](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234137052.png)

![image-20200923234142651](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20200923234142651.png)

## Conv2D参数

### input_shape

作为第一个层时，必须设置输入形状（忽略样本维度），例如input_shape=(128, 128, 3) for 128x128 表示128*128的RGB图片。

### filters

卷积过滤器数量，int类型，必须，例如32，表示生成32个特征图

### kernel_size

卷积窗口大小，int/tuple类型，必须，例如(3,3)表示宽为3高为3的滑动窗口，只传一个整数时表示宽和高都一样。

### strides

卷积窗口移动步进，int/tuple类型，默认为(1,1)表示每次只滑动1个单元，只传一个整数表示宽和高的步进都一样，当取值不为1时，dilation_rate必须为1。

### padding

取值为valid或者same，valid表示只取有效值， same表示填充以和原图保持相同大小。

### data_format

输入数据格式，取值为channels_last或者channels_first。默认值为channels_last。
channels_last： 输入shape为(batch, height, width, channels)
channels_first：输入shape为(batch, channels, height, width)

### dilation_rate

空洞卷积比例，int/tuple类型，默认值为(1,1)，当取值不为1时，strides必须为1。

### activation

激活函数，不传时默认为a(x)=x

### use_bias

是否使用偏置向量，默认为True

### kernel_initializer

卷积核初始化器，默认为glorot_uniform

### bias_initializer

偏置向量初始化器，默认为zeros

### kernel_regularizer

卷积核正则化函数

### bias_regularizer

偏置向量正则化函数

### activity_regularizer

输出结果正则化函数

### kernel_constraint

卷积核约束函数

### bias_constraint

偏置向量约束函数



## Maxpooling2D

### pool_size

下采样比例，int/ tuple类型, 例如取值为(2,2)时，输入为(s,32,32,c)，那么输出为(s,16,16,c)。

### strides

步进值，int/tuple, 或者None.如果取值为None, 步进默认等于pool_size。

### padding

取值为valid或者same

### data_format

输入数据格式，取值为channels_last (默认)或者 channels_first



## 详细

**CNN**是一种通过卷积计算的前馈神经网络，其是受生物学上的感受野机制提出的，具有平移不变性，使用卷积核，最大的应用了局部信息，保留了平面结构信息。



git更新：git clone https://github.com/git/git

## 天池AI学习

### 神经网络（上）

#### 人工神经元

![image-20210904012815809](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904012815809.png)

输入为离散的，即0或1

输入→权重→求和→激活函数→输出

y=wx+b中：权重即是w，偏置即是b。这个表达式就是单层感知机的形式

![image-20210904011902966](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904011902966.png)

x1和x2是输入，表格中代表输出

在逻辑与中：权重都是2，偏置是3

![image-20210904012438322](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904012438322.png)

在此图中，x1和x2的权重都是1，但是偏置不同，单层感知机不能实现异或操作

#### 单层感知机

![image-20210904013202304](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904013202304.png)

输入可以是浮点型

比如sigmoid函数就是这样的结构

此图中的数学表达式：权重→求和→减去偏置→激活函数

![image-20210904014813995](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904014813995.png)

图中的数学表达式即为梯度下降法的表达式

通过这个准则去更新x，对于感知机来说，表达式中的x就是权重（w）

Wt表示t时刻的权重；Wt+1表示下一步的权重

学习率用于控制更新的步长，右下角两幅图的学习率分别是0.26和0.1，0.26的步长太大，寻找极值的效果差，每一次更新幅度大，要寻找合适的学习率

![image-20210904020223435](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904020223435.png)

要把两种颜色分开，数据是三维坐标，一维是偏置，二维是x坐标，三维是y坐标；为输入数据的格式，数据标签分别为1和-1→y=0为分界线

需要划一条线来把数据分开

![image-20210904020818031](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904020818031.png)

L为LOSS函数，此时L即为优化目标

Δw表示当前要更新的值

全局变量中：X为输入，Y为标签，W为权重，lr为学习率，n为迭代次数

有两条线都能分割数据→用单层感知机来解决这类问题的时候答案是多个的，不是唯一的，有多个正确答案

给W不同的初始值，优化的时候就可能得到不同的优化结果



![image-20210904022333557](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210904022333557.png)

无论怎么划，都不能有一条直线能把左图的数据给正确划分，要增加神经网络的深度，增加更多的非线性操作才能实现异或操作。

### 神经网络（下）

![image-20210909094951866](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909094951866.png)

在隐藏层中对输入进行非线性处理，再对其用激活函数，能实现更多的非线性操作

![image-20210909095723933](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909095723933.png)

对比线性分界面与非线性分界面，非线性在式子上加了一个σ（非线性激活函数），即o与x之前有一个σ函数，若中间有更多隐藏层，则可以有更多非线性的变化，非线性变化实质上是对数据从一个空间向另一个空间转换

![image-20210909152434284](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909152434284.png)

![image-20210909152549215](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909152549215.png)

前向传播里传的是权重，而反向传播传的是梯度，反向传播的起始层是损失函数，图右绿圈是真值，黑圈是预测值

![image-20210909153137612](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909153137612.png)

右上角的误差算法为C（损失函数）对Xjl的求导

![image-20210909161742491](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909161742491.png)

![image-20210909163333826](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909163333826.png)

反向传播算法能使误差最快达到最小

![image-20210909164018191](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909164018191.png)

可见层也是数据层，通用的玻尔兹曼机即使是层与层之间的神经节点也有连接

在玻尔兹曼机中，并没有明确数据输入和输出的方向性，传播方向不确定

RNN和递归神经网络其实就是非前馈神经网络

![image-20210909164751131](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909164751131.png)

![image-20210909165105235](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909165105235.png)

![image-20210909165527683](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210909165527683.png)

如softmax所示，如果直接把图片作为输入，而在图片的像素中，一个像素周围相似的像素会与它的连接更紧密，但是离的远一点的连接就不那么紧密

那么这就无法区分它的结构特点，因为这个神经网络是全连接的，它无法区分局部的结构关系

而卷积比全连接层多了通过卷积对图像局部的特征求取

### 卷积神经网络（上）

![image-20210910231128480](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910231128480.png)

刺激兴奋区域会使细胞兴奋，刺激抑制区域会使细胞处于抑制状态，感受野是局部的，不能感受到过远的信号

![image-20210910231717141](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910231717141.png)![image-20210910231717436](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910231717436.png

![image-20210910232543542](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910232543542.png)

![image-20210910232941191](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910232941191.png)

S层进行特征提取，C层对S层的结果进行进一步的抽象，对S层的结果进行池化操作，获得一个更小的更抽象的特征

![image-20210910233425673](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910233425673.png)

不断通过网络加深，不断抽象和增加它的感知

![image-20210910233756839](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910233756839.png)

在卷积神经网络中，不需要人工进行特征提取

![image-20210910234102932](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910234102932.png)

蓝色表示输入，绿色表示输出，有个3x3的卷积核在滑动，再得到绿色的输出

![image-20210910234550724](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910234550724.png)

灰度图有一个通道，RGB有三个通道，RGBA还有一个透明的通道，有四个通道。通道越多，表达的特征越多

![image-20210910234720225](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910234720225.png)

举例如左上角的3x3与中间的3x3中的每个数对应起来相乘再对这九个结果进行相加，最后得到右边的输出（左上角的4）

卷积核无论如何滑动，大小都一样，而且权重共享

![image-20210910235340415](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210910235340415.png)

![image-20210913225025180](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913225025180.png)

输出特征图中有Co个通道，其中每一个通道都与输入通道中的每个通道进行连接，而要与相同数量的卷积核对应卷积计算，所以卷积核的数量应该是Ci x Co

![image-20210913230016505](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913230016505.png)

若对输入为3x3的图像进行2x2的滑块进行卷积，那么就会有2x2的输出，但经过padding层后在3x3的图像外面加多了一圈再进行2x2的卷积，就会得到4x4的输出，比原始的输出要大



这样做更多是为了防止图像的分辨率发生变化

![image-20210913230432108](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913230432108.png)

之前是默认步长为1，也就是挨个滑动

此处的卷积是3x3，左图步长为1挨个滑动

stride过大很多时候是指步长比卷积核还要的宽还要大的情况，容易造成信息遗漏

![image-20210913231126110](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913231126110.png)

卷积核大小表示为k*k，在这里表示为它的一边，用于输出大小的计算

![image-20210913231450371](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913231450371.png)

Max pool：如左上角的4x4中则取6（即最大数）

Avg pool：如左上角的4x4则取四个数的平均数也就是3

这样池化可以降低分辨率，减小算度，压缩特征进行抽象，因为很多时候对图像并不需要关心太多的细节

![image-20210913232730934](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913232730934.png)

Conv1中的四个3是经由Raw Image中的3x3卷积得到的，而Conv2中的5则是对应了Conv1中的数据，感受到了Raw Image中的5x5大小，所以它的感受野大小是5x5

步长的大小对感受野的大小也是一个很重要的影响因素，感受野的计算方法对于不同的神经网络来说是比较复杂的，通常来说要用递归的方法进行计算，越深层的像素感受野是要基于前面一层的感受野计算然后才能得到

![image-20210913233446116](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913233446116.png)

一般来说，卷积大小都会比图像输入大小要小

离的越远的像素之间的连接会更松，所以卷积神经网络的方法是合理的

图像上局部与整体本身就有统计自相似性，有的时候规律可通过图像的局部的块就能够进行表征，那么在这个时候就没有必要对整个图像进行处理，只需要对一个小的图像块就能够完成一些功能，比如提取人脸，或者提取一些可辨识性的特征，故局部连接是很适合做图像处理的结构

![image-20210913234238935](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210913234238935.png)

如左图中的足球，若权重不共享则对足球进行迁移的时候图像就会发生变化，就不能得到黑色背景中的足球，而权重共享则能保证信息不变

同时还能使卷积逐个扫描不重叠，移动方式不重叠，能够进行局部连接，能降低算度，相对于全连接降低了4哥数量级，这样才能构建一个更复杂的模型，才能对其进行继续优化

### 卷积神经网络（下）

![image-20210914222415054](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914222415054.png)

##### 全连接层中的反向传播算法

概括就是第L层的神经元j的误差值等于L+1层所有与神经元j相连的神经元他们之间的误差值的加权

式子中的第二个符号是第L+1层神经元j的误差值

##### 卷积神经网络中的反向传播算法

![image-20210914224932881](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914224932881.png)

![image-20210914231801112](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914231801112.png)

*以后回来看

![image-20210914231938182](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914231938182.png)

最大池化：后记下了最大值的位置，进行反向传播的时候对其他位置填充为0，故计算误差时最大值才有影响

平均池化：如将3除4得到0.75，则0.75则是每个元素的误差



自动微分能自动反向传播，不需要手动进行反向传播

![image-20210914233351415](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914233351415.png)

![image-20210914233612248](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914233612248.png)

![image-20210914234203353](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914234203353.png)

![image-20210914234246679](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914234246679.png)

这些技巧都不是我们现在的卷积神经网络会用的技巧，只是当时背景所需要用

![image-20210914234401317](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914234401317.png)

![image-20210914235057112](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210914235057112.png)

AlexNets用了两个GPU通道进行训练，是因为当时的GPU还不够好

不重叠池化指步长与卷积核大小相等

### 激活函数与参数初始化

人脑细胞激活特征

![image-20210918223327573](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210918223327573.png)

![image-20210919074335562](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919074335562.png)

若x为输入到y，y再输出到z，而y和z之间若没有非线性的激活函数，那即使z与y的表达式复杂一些，但仍然是线性的

![image-20210919074924753](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919074924753.png)

Sigmoid函数：没有负值激活，也就是其输出的激活值的值域都是正的，神经元都是正的激活层

![image-20210919075417120](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919075417120.png)

relu也无负的激活值

![image-20210919075623745](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919075623745.png)

根据α的值分为上述三种激活函数，L是固定的，P是根据不同网络层或通道学习而选择α的，P是随机选择（如从均匀分布中取一个随机值）的

PReLU：在浅层特征更大更稠密，而到了深层α值较小代表负区间的激活值更小，即激活的幅度更小，也更类似与原始的ReLU模型

![image-20210919081500692](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919081500692.png)

![image-20210919081641245](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919081641245.png)

![image-20210919082929949](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919082929949.png)

![image-20210919083153108](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919083153108.png)

GELU可看作Swish函数的变种

![image-20210919191351279](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919191351279.png)

利用无监督学习预训练后用反向传播算法对多层前馈神经网络进行微调，这样更稳定，长时间都作为模型训练的范式，解决了难以训练的问题



##### 无监督学习的例子

![image-20210919191610985](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919191610985.png)

![image-20210919191634981](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919191634981.png)

![image-20210919191653636](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919191653636.png)

![image-20210919192347322](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919192347322.png)

上面为激活值，下面为梯度值

![image-20210919192654796](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919192654796.png)

![image-20210919195322365](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919195322365.png)

固定每一层的参数的方差，此时均匀分布的上限和下限是a和b，Var（w）即指方差，然后再通过均匀分布，将参数的分布固定在[-r,r]之间，此时参数分布的上限和下线是r和-r，但网络层数越深的时候激活值会越来越小，只是参数的方差被固定，但输入和输出并没有固定，故配合对特征的激活值进行归一化

![image-20210919201116839](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919201116839.png)

sigmoid和tanh都满足Glorot的条件，这样能比较好的优化神经网络,一开始是对tanh来设计的

前向传播中，y的方差等于wx激活的形式对其求方差，若要对y的方差与x的方差保持一致，则需要满足n乘以x的方差=1

反向传播中，ni代表当前层神经元的数量，ni+1代表下一层神经元的数量，上图式子仍在使用均匀分布，b和a分别代表均匀分布的上限和下限

最后的式子表示根据神经元数量自动调整初始化方差分布，W是方差，Var（）就表示括号里是一个变量

此初始化方法应用最广泛

![image-20210919203847251](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210919203847251.png)

右图是常数和权重更新幅度的关系，差不多在1.4能取得比较稳定的关系

对一些大型的网络一般会用预训练好的网络来进行初始化，但这也需要一个好的初始化方法



##### 初始化的概念

参数初始化 又称为 权重初始化 （weight initialization）或 权值初始化 。 深度学习模型训练过程的本质是对weight（即参数 W）进行更新，这需要每个参数有相应的初始值。 说白了，神经网络其实就是对权重参数w不停地迭代更新，以达到较好的性能。 模型权重的初始化对于网络的训练很重要，不好的初始化参数会导致梯度传播问题，降低训练速度；而好的初始化参数能够加速收敛，并且更可能找到较优解。 如果权重一开始很小，信号到达最后也会很小；如果权重一开始很大，信号到达最后也会很大。



### 标准化与池化

![image-20210920112121658](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920112121658.png)

标准化/归一化 其实是一种数据预处理的方法

零均值归一化就是减去均值再除以方差，将其归一化成一个标准的正态分布

直方图均衡化中，cdp（rk）是累计概率分布

右下图就是经过直方图均衡化得到的结果，使亮度分布更加均匀，上图的亮度分布只在0-100之间

归一化可使图像的辨识度更高,使数据更加有效

![image-20210920113954084](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920113954084.png)

去除量纲的干扰：比如例子中的30和170和1和500000的数量级差距很大，若不归一化则训练的时候小的特征会被淹没

保证数据有效：右图红色的是梯度，防止梯度消失等问题

![image-20210920163128541](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920163128541.png)

batch size就是每次训练的样本，c是通道数

现没有c这个维度，每个维度单独进行计算

让其进行归一化之后再乘**γ**再加**β**，这样就能调整数据分布跳出线性区域

![image-20210920202749932](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920202749932.png)

![image-20210920204411143](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920204411143.png)

batch normoalization在每一个batch中都会计算他的均值和方差去进行归一化，但在训练的时候是用已经训练好的均值和方差来训练，尤其是batch越小的时候差异性越大，而batch renormalization是在对每一个batch进行均值和方差的计算时还要对它进行调整

例如在计算完xi之后还要进行对其乘以r和一个d，上图中d和r的表达式贴的比较近，其实是分开的

BN层其实就是batch normalization层



![image-20210920205647088](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920205647088.png)

LN层是对层进行归一化，把不同的通道拿到一起进行归一化，但对不同样本是进行单独计算的，它适合非定长输入，因为可以摆脱batch参数

GN层可看作LN层的变种，G代表对通道进行分组

IN层中H*W代表对每一个特征通道、样本都单独进行计算

![image-20210920210708671](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920210708671.png)

SN层其实就是对不同的标准化方法的结合

不同的batch size也适用于不同的标准化方法

![image-20210920211156460](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920211156460.png)

对于平移和旋转不变性是有局限性的，随着神经网络的层数加深他们会变弱，不过在一定程度上可以获得平移旋转不变性

![image-20210920211507324](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920211507324.png)

最大池化能将一些比较明显的纹理特征保留下来，将一些比较平滑的信息抑制掉

![image-20210920211851537](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920211851537.png)

经过激活函数之后，基于它的激活值，要对他进行归一化，再进行概率分布的计算，有一定的随机性，但元素被选中的概率与数值大小成正相关，并不绝对，有一定的概率

![image-20210920212225032](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920212225032.png)

![image-20210920212421968](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210920212421968.png)

学习完不管什么池化方法都差不多，甚至是把池化操作融到卷积里面，变成带步长的卷积操作来替代它也可以有这样的效果，只不过它需要学习参数而池化不需要学习参数

所以池化并不是必要的，通常可以用带步长的卷积来代替池化的操作



### 泛化与正则化

![image-20210923195917279](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923195917279.png)

左图中红色的曲线表示泛化误差（测试集的误差），蓝色的曲线表示的是训练集上的误差，欠拟合指没有得到很好的训练，无法拟合此数据集，此时模型的表达能力还不行。



右图中第一幅图表示欠拟合状态，中间的图表示是比较好的状态，模型的复杂度得到比较好的控制，是一条平滑的曲线，在这样的状态下，不可避免会有一些样本没有被正确划分，最右图表示过拟合，是一条非常复杂的分界面，确实可以使得所有样本都能得到正确的划分，但这样的曲线过于复杂，这样的模型去面对一些新的数据是时表现就会比较差。

要有比较好的泛化能力，就要让它处于欠拟合和过拟合之间



![image-20210923200945615](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923200945615.png)

泛化能力不好时：

如左1图，更改了像素值便有75%的概率将狗认成猫，

中间图人加了贴纸便不能识别，右图同理

![image-20210923201203923](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923201203923.png)

V为损失函数，也就是所谓的经验风险

显式正则化也就是通过显式的方法去控制模型，去直接控制它的参数，控制它的优化目标

隐式正则化是通过一些其他的方法来间接影响模型的泛化能力

![image-20210923201844957](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923201844957.png)

应为测试集误差开始持续增大

![image-20210923202217354](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923202217354.png)	第二种方法的模型可以采用不同的结构和不同的超参数

![image-20210923202716573](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923202716573.png)

训练时神经元是否存在的概率为p，但在测试时所有神经元均存在，所以在测试时需要给模型的权重乘以p，2的n次方表示任何两个神经元连接起来的模型集合起来

![image-20210923203406820](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923203406820.png)

有更多接近于0的激活值，与人脑的激活模式更相似

![image-20210923203717220](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923203717220.png)

左下图是随机丢弃一些值的单元，打叉的部分表示丢弃掉的单元，并且把它们变成恒等映射

![image-20210923204601427](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923204601427.png)

左图是L2正则化，右图是L1正则化

L1正则化函数与损失函数的交点在坐标轴上，此坐标图有两个参数，即两个权重，在坐标轴上的交点即表示有一些权重为0，L1正则化的特点就是会使一些参数为0，获得参数的稀疏化

L2的最优交点不会在坐标轴上，所以会获得比较小的一些参数，也能获得类似的稀疏化

L1和L2正则化都能控制参数

权重衰减和正则化都是等价的，都能控制参数

![image-20210923205509088](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210923205509088.png)

数据增强对泛化能力的增强是最有效的

随机梯度下降法，给同一个模型同样的参数，不同次训练结果不同	

标签噪声指故意人为地给样本的标签一些错误的值，或给标签一些扰动；如给0以1/p的概率

标签有一些噪声就可以抑制过拟合，也有正则化的效果

很多时候隐式正则化都能比显式正则化有更好的效果

### 最优化

![image-20211005155512420](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005155512420.png)

最优化目标是loss函数并找到极值点使其最小

![image-20211005160546780](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005160546780.png)

对于深度学习模型来说，包含很多局部极小值，则最优化的目标就是获得不错的局部最小值，优化起来有一定的难度

![image-20211005161343201](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005161343201.png)

鞍点是最优化中比较常见的问题，尤其是要注意优化鞍点方向的调整

![image-20211005162745231](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005162745231.png)

学习率就是乘在梯度前面的一个乘因子，学习率越大，参数更新的幅度越大

在左图的展示中，就有不同的学习率衰减的策略，就像蓝色的曲线，就代表了学习率较小，而在学习率较小的情况下，深度学习模型收敛的时间会比较长甚至不能收敛，故需要谨慎地选择学习率



现在有许多深度学习优化方法，主要是一节优化方法，在其中，有基于更新方向的，也有基于选择更为合理的学习率的

![image-20211005171426710](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005171426710.png)

在上面的部分写了梯度下降法的式子，按着梯度的反方向来进行搜索，如在极值点的左边就往右搜索，在右边就往左搜索，满足梯度下降法的式子即可

下图中的式子表示下一步的参数=这一步的参数—它的学习率乘它的梯度

![image-20211005171949253](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005171949253.png)

在右图中黑色表示SGD原来的样子，黑色的箭头表示SGD在优化时走的弯路，但用动量法（红色箭头），就可以少走很多弯路，，更快找到最优点

而在计算式子中，相比原来的SGD，在这里多了一个βmn，这个βmn就是指数的衰减



在左下图中，由A点沿着A点的梯度方向移动到B点，对于正常的梯度下降算法则是在B点沿着B点的梯度方向更新就可以了，而对于动量法，它要在B点累加上动量项（A点到B点的梯度项），也就是βmn，再与B点的梯度方向合并起来，作为B点真正的梯度更新方向。

这是最常见的SGD算法的改进

![image-20211005173132572](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005173132572.png)

NGA法在动量法中添加了一个校正因子，算法如图理解即可，NAG算法效果不一定比动量法要好

![image-20211005174010978](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005174010978.png)

左边式子中的分母作为权重因子，Δθt是参数的更新幅度

![image-20211005180211442](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005180211442.png)

RMSprop可以一定程度上缓解到了后期学习率较小的问题，它的Gt不是对所有时间段的梯度都进行了累加，只累加了一个窗口的梯度，并且使用了动量平均计算

![image-20211005182139950](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005182139950.png)

Adam算法算是前面几个算法的融合，但也存在缺点，如图

![image-20211005184330601](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005184330601.png)

SGD的改进算法不一定更好，优点在于减少了调参工作量，但缺点在于不稳定，收敛效果不如精细调参的SGD算法

![image-20211005200842248](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005200842248.png)
