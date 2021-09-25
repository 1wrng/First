# 人工智能数学基础

![image-20210925195727752](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925195727752.png)

### 导数

![image-20210925201123726](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925201123726.png)

右图两种分别是x的绝对值函数和relu激活函数在0点时均不可导，因为左导数与右导数不等

### 泰勒展开

![image-20210925202618581](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925202618581.png)

f的后续为n阶导



![image-20210925203122730](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925203122730.png)

列向量就是竖着来的，行向量就是横着来的，有个转置T能使列向量变为行向量，使行向量变为列向量

![image-20210925203445060](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925203445060.png)

XT乘Y等于XY两个向量对应的元素相乘再加起来

![image-20210925203923748](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925203923748.png)

二范数就是模，单位向量就是模为1

![image-20210925204243881](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925204243881.png)

方针、对称矩阵、单位矩阵 都是行列数相等的

对称矩阵关于对角线对称，单位矩阵的对角线上都是1，其他都为0，简称为I，作用等同于数里的1



![image-20210925222626064](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925222626064.png)

![image-20210925222919984](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925222919984.png)

乘法就是各矩阵中的行和列分别相乘再累加



![image-20210925224410319](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925224410319.png)

矩阵乘法不满足交换律，转置则是后转置再乘前转置

![image-20210925225717513](C:\Users\1wrng\AppData\Roaming\Typora\typora-user-images\image-20210925225717513.png)矩阵中是没有除法的，AB=I中的A是方阵，I是单位矩阵，则AB中B为右逆，BA中B为左逆，如果这样的B存在的话，左逆和右逆一定都相等，统称为A的逆或A的非