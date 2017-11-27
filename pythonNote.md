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

### np.split() np.hsplit() np.vsplit()



```python
split(ary, indices_or_sections, axis=0)
#Split an array into multiple sub-arrays.
#np.hsplit:  split(ary, indices_or_sections, axis=1) along second axis
#np.vsplit: split(ary,indices_or_sections, axis=0) along first axis
```

```python
>>> x = np.arange(9.0)
>>> np.split(x, 3)
[array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]
    
>>> x = np.arange(8.0)
>>> np.split(x, [3, 5, 6, 10])
[array([ 0.,  1.,  2.]),
array([ 3.,  4.]),
array([ 5.]),
array([ 6.,  7.]),
array([], dtype=float64)]
```

### np.reshape()

```python
# in row order
>>> a = np.arange(6).reshape((3, 2))
>>> a
array([[0, 1],
	[2, 3],
    [4, 5]])
```

### np.repeat()

注意和tile的区别

```python
    >>> x = np.array([[1,2],[3,4]])
    >>> np.repeat(x, 2)
    # return flatten
    array([1, 1, 2, 2, 3, 3, 4, 4])
    # repeat along second axis
    >>> np.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]]
    # repeat along first axis
    >>> np.repeat(x,2,0)
    array([[1, 2],
           [1, 2],
           [3, 4],
           [3, 4]])
    >>> np.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])
```

### np.vstack()

```python
	>>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.vstack((a,b))
    array([[1, 2, 3],
           [2, 3, 4]])
```

### np.hstack()

```python
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.hstack((a,b))
    array([1, 2, 3, 2, 3, 4])
    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[2],[3],[4]])
    >>> np.hstack((a,b))
    array([[1, 2],
           [2, 3],
           [3, 4]])
```

