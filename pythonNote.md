# Python Note

### tile

```python
>>> import numpy  
>>> numpy.tile([0,0],5)#在列方向上重复[0,0]5次，默认行1次  
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  
>>> numpy.tile([0,0],(1,1))#在列方向上重复[0,0]1次，行1次  
array([[0, 0]])  
>>> numpy.tile([0,0],(2,1))#在列方向上重复[0,0]1次，行2次  
array([[0, 0],  
       [0, 0]])  
>>> numpy.tile([0,0],(3,1))  
array([[0, 0],  
       [0, 0],  
       [0, 0]])  
>>> numpy.tile([0,0],(1,3))#在列方向上重复[0,0]3次，行1次  
array([[0, 0, 0, 0, 0, 0]])  
>>> numpy.tile([0,0],(2,3))<span style="font-family: Arial, Helvetica, sans-serif;">#在列方向上重复[0,0]3次，行2次</span>  
array([[0, 0, 0, 0, 0, 0],  
       [0, 0, 0, 0, 0, 0]])  
```

### python--sum函数--sum(axis=1)

平时用的sum应该是默认的axis=0 就是普通的相加，当加入axis=1以后就是将一个矩阵的每一行向量相加。

例如：

```
1 >>>import numpy as np
3 >>>np.sum([[0,1,2],[2,1,3],axis=1)
5 array（[3,6]）
```

```
1 c = np.array([[0, 2, 1], [3, 5, 6], [0, 1, 1]])
2 print c.sum()
3 print c.sum(axis=0)
4 print c.sum(axis=1)
5 结果分别是：19, [3 8 8], [ 3 14  2]
6 axis=0, 表示列。
7 axis=1, 表示行。
```
### python正则表达式 re compile

compile(pattern, flags=0) 
Compile a regular expression pattern, returning a pattern object.

Doc: http://www.cnblogs.com/huxi/archive/2010/07/04/1771073.html

```python
# encoding: UTF-8
import re
 
# 将正则表达式编译成Pattern对象
pattern = re.compile(r'hello')
 
# 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
match = pattern.match('hello world!')
 
if match:
    # 使用Match获得分组信息
    print match.group()
 
### 输出 ###
# hello
```

### python strip()

返回移除字符串头尾指定字符后重新生成的新字符串