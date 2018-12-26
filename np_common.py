import numpy as np
## random seed
np.random.seed(3) ## only integer;


'''


1. full
2. astype

'''



### 1. full;    like tf.fill

pad_value = 2
a = np.full((2,3,4), pad_value)
print(a)
print(a.shape)


### 2. astype; change dtype
a = np.random.random((3,4))
b = a.astype(int)
print(b)



