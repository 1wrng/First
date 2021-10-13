# C

#### 小甲鱼

##### 打印

多一个反斜杠的作用

![image-20211004120850863](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004120850863.png)

```c
#include <stdio.h>

int main()
{
	printf("hello world\n\
111");
	return 0;
}
```

和

```c
#include <stdio.h>

int main()
{
	printf("hello world\n111");
	return 0;
}
```

的效果是一样的

#### 变量

![image-20211004121621344](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004121621344.png)

​	

![image-20211004121813393](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004121813393.png)

![image-20211004121939032](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004121939032.png)

![image-20211004122354344](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004122354344.png)

字符型表示一个字节，一个字符则用单引号，多个字符称字符串则为双引号

%d表示十进制显示a然后换行

%c表示在那个位置占了一个位置，然后把b转换成%c也就是字符的形式然后放到该位置

%.2f表示小数点后两位，%11.9表示d的数据总占宽11位，小数点后9位

#### 常量和宏定义

![image-20211004123034334](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004123034334.png)

![image-20211004123152914](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004123152914.png)

![image-20211004200059307](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004200059307.png)

宏定义不需要在后面加分号，URL和NAME等都是符号常量，一般用全部大写来命名符号常量，小写字母来命名变量



标识符就是C语言中所有的名字，规则和变量的命名规则是一样的

![image-20211004200706300](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004200706300.png)

在字符串的结尾C会自动加一个\0来代表字符串的结束，所以字符串的长度读取出来会多一个

#### 数据类型

![image-20211004202102193](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004202102193.png)

![image-20211004202628301](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004202628301.png)

类型是int等数据类型，int占4个字节，理论上存放32个1

![image-20211004203950392](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004203950392.png)

中括号内的内容可写可不写，signed带符号位，unsigned不带符号位

%d打印带符号位的，%u打印不带符号位的

#### 取值范围

CPU能读懂的最小单位是比特位-bit/b

内存机构的最小寻址单位是字节-Byte/B 一个字节=8位

当二进制所有位都是1时转换为十进制就是2的n次方-1，n就是二进制的位数



在默认情况下int都是signed类型的也就是带符号的![image-20211005204910465](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005204910465.png)

故int中不能放2的32次方-1，unsigned int 才能放2的32次方-1，int中只能存放2的31次方-1，因为左边第一位是符号位，符号位不表示值

![image-20211005205111990](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005205111990.png)

按位取反就是把1变成0，把0变成1（除了符号位）

![image-20211005205503779](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005205503779.png)

![image-20211005205638550](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005205638550.png)

比如-2来说，到了第二步的-2应该是1 1 1 1 1 1 0 1，但加了1之后进了一位，就变成了1 1 1 1 1 1 1 0，正数需要多一个地方来表示0，而负数不需要，所以负数可以到-128而正数只能到127

![image-20211005205916830](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005205916830.png)

![image-20211005210158601](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211005210158601.png)

#### 字符和字符串

用char来存放整数时

```c
#include <stdio.h>

int main()
{   
	char a = 'C';
	printf("%c=%d", a, a);
	return 0;
}
```

这样显示出来的结果是 c=67，因为C对应的ASCII表中的67

而若代码是

```c
#include <stdio.h>

int main()
{   
	char a = 70,b=105,c=115,d=104,e=67;
	printf("%c%c%c%c%c",a,b,c,d,e);
	return 0;
}
```

得到的打印结果是FishC，%c是单个字符输出的意思

故当声明为字符变量char，而输入了整数，在输出时表明了是%c，便会在ASCII中寻找对应的字符并打印出来



```c
#include <stdio.h>

int main()
{   
	char a = 170;
	printf("%d",a);
	return 0;
}
```

此时会输出-86，因为char 默认是signed，取值范围在-128~127，而170-127=43，128-43=85，因为是翻过来所以要+1得到86，结果就是-86



对于字符串，可以

```c
char a[5];
```

则表示a中字符的数量

变量名[索引号]=字符

如承接上面的

```c
a[0]='b';
a[1]='c';
a[2]='d';
a[3]='e';
a[4]='f';
```

或用更简单的方式

```c
char a[5]={'b','c','d','e','f'};
```

打印字符串用的是%s

```c
#include <stdio.h>

int main()
{
	char a[5] = { 'b','c','d','e','f'};
	printf("%s", a);
	return 0;
}
```

这样写的话打印出来的bcdef后面会有乱码，因为没有字符串结束标志，也就是\0,

所以

```c
#include <stdio.h>

int main()
{
	char a[6] = { 'b','c','d','e','f','\0'};
	printf("%s", a);
	return 0;
}
```

这样才能正常打印

同时char a[6]的6是可以不写的，就直接char a[]，系统会自动帮你计算有几个字符

或者

```c
    char a[]={"bcedf"};
```

则不用去在后面加\0，因为这是字符串常量，系统会帮你自动添加，甚至{}都可以不要

就是‘

```c
char a[]="bcedf";
```

#### 算术运算符

![image-20211007194717258](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211007194717258.png)

两个整数相除会舍弃小数，而若两个浮点数相除得到的也是浮点数，会保留小数后6位

求余运算只能是整数，不能是浮点数

其中双目指的是有两个操作数，比如5+3那么5和3都是操作数，+则是算数运算符



如果不同类型的数据相加或进行其他运算类型，会将占位较小的数据类型转换为占位较大的数据类型再进行操作

```c
#include <stdio.h>

int main()
{
    
	printf("%d\n",1+2.0);
    printf("%f",1+2.0);
	return 0;
}
```

第一个结果是0，第二个结果是3，第一个会错是因为应该输出的结果是浮点型，不是整型

除非将

```c
	printf("%d\n",1+(int)2.0);
```

将浮点型强制转换为整型的话是会往小的取，如(int)1.8得到的是1，相当于把小数位给去掉.

#### 关系运算符和逻辑运算符

![image-20211008205305920](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211008205305920.png)

如果对字符进行加减等运算时，会将字符对为ASCII表中对应的数

```c
#include <stdio.h>

int main()
{
	printf("%d\n",'a'+'b'>='c');
    return 0;
}
```

得到的结果是1，表示正确，0则表示错误

![image-20211008210910241](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211008210910241.png)

单目运算符的优先级都比双目运算符的优先级高

![image-20211008211922267](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211008211922267.png)

在第二个表达式中，3+1应该=4，而在逻辑运算符的两边输入任何非0的数都表示真，也就是1

在第四个表达式中，！0=1，而1+1<1不成立，所以左边得到的值是0，右边也是0，整个表达式的值就是0

![image-20211008212708127](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211008212708127.png)

比如

```c
#include <stdio.h>

int main()
{
	int a=3,b=3;
    (a=0)&&(b=5);
    printf("%d,%d\n",a,b);
    (a=1)||(b=5);
    printf("%d,%d\n",a,b);
	return 0;
}
```

得到的结果分别是0，3和1，3

在第一个表达式中，因为是逻辑与，a被赋值为0，那么就直接判断整个结果表达式为0，不会进行右边b=5的赋值，所以b还是3

在第二个表达式中同理，第一个已经被赋值为1，那么整个结果判断为1，不会进行右边的计算

#### If语句

![image-20211010093333092](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211010093333092.png)

同样表达式填入非0即为真，0即为假

当执行语句只有一个语句的时候直接在括号后面接就可以了，不一定要大括号；

![image-20211010095148696](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211010095148696.png)

![image-20211010095213092](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211010095213092.png)

#### Switch case

![image-20211011211142654](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211011211142654.png)

default是当表达式中的值表达的在 case 中都没有找到符合的值，那么就执行default中的语句或者程序块

当然default是可选的，不写的话而且case中没有找到符合的值switch case 不会执行任何语句

表达式中是常量或者常量表达式

如果在case后面不加break，即使满足了当前的case执行此式子，也会继续往下执行别的case语句，如表达式满足case2，就会执行case2的语句，如果没有break，就会一直往下执行



scanf里面一般双引号只存放占位符

getchar（）就是从标准输入中读取下一个字符，如有两个scanf，而输入中间隔着一个空格，就需要一个getchar



if else中的else是和最接近它的if连在一起的，是和空格和TAB没关系的

而要摆脱这种情况可以

```c
if()
{	
 	if() }
else
{
    
}
```

即可解决

\a是发出警报

%e是使数以指数形式输出，%A是以十六进制方法输出（p方法）

整型和浮点型不能直接运算
