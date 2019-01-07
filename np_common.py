import numpy as np
## random seed
np.random.seed(3) ## only integer;


'''


1. full
2. astype
3. tensordot
'''



### 1. full;    like tf.fill
'''
pad_value = 2
a = np.full((2,3,4), pad_value)
print(a)
print(a.shape)
'''

### 2. astype; change dtype  like tf.cast 
'''
a = np.random.random((3,4))
b = a.astype(int)
print(b)
'''


### 3. tensordot

##  reference:  https://www.machenxiao.com/blog/tensordot

a = np.ones([5, 4, 2, 3])
b = np.ones([3, 2, 6])
assert np.tensordot(a, b, (2, 1)).shape == (5, 4, 3, 3, 6)
assert np.tensordot(a, b, (3, 0)).shape == (5, 4, 2, 2, 6)
assert np.tensordot(a, b, ((2, 3), (1, 0))).shape == (5, 4, 6)
assert np.tensordot(a, b, ((-2, -1), (1, 0))).shape == (5, 4, 6)


a = np.ones([5, 4, 3, 2])
b = np.ones([3, 2, 6])
assert np.tensordot(a, b, axes = 2).shape == (5, 4, 6)



