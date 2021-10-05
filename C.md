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

类型是int等数据类型

![image-20211004203950392](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20211004203950392.png)

中括号内的内容可写可不写，signed带符号位，unsigned不带符号位

%d打印带符号位的，%u打印不带符号位的

#### 取值范围

CPU能读懂的最小单位是比特位-bit/b

内存机构的最小寻址单位是字节-Byte/B 一个字节=8位

当二进制所有位都是1时转换为十进制就是2的n次方-1，n就是二进制的位数