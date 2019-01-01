import tensorflow as tf
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(3)
np.random.seed(1)
tf.set_random_seed(5)

""" 
1. tf.multiply, tf.matmul, 
2. tf.cast 
3. tf.zeros / ones / zeros_like / ones_like 
4. tf.constant & tf.Variable 
5. random_normal & random_uniform & random_shuffle 
6. tf.argmax  argmin 
7. tf.gather  tf.gather_nd *********  indexing 
8. tf.maximum & minimum;  greater & less & equal
9. tf.add/div  log/exp
10. concat & stack & unstack  ##增加维度stack 
11. tf.expand_dims & tf.squeeze
12. reshape & shape & get_shape(can't use run) *******  & set_shape (different from reshape) & size ****** ### dynamic shape, static shape 
13. transpose 
14. tf.tile   ## 堆叠 replicate
15. reduce_sum reduce_mean 
16. fill
17. where 		2 usages
18. assign
19. to_float  to_int  "cast" into float 
20. TensorShape 
21. clip_by_value
22. placeholder
23. tf.norm  matrix norm 
24. tf.nn.sigmoid_cross_entropy_with_logits
25. tf.nn.softmax_cross_entropy_with_logits
26. embedding_lookup
27. tf.nn.conv1d & tf.layers.conv1d
28. tensor => np.array
29. tf dtype

### list of list; numpy.array can be used as input for TF 


advanced topic
	Variable  get_variable
	variable_scope
	## Tensorflow踩坑记  https://blog.csdn.net/hejunqing14/article/details/52688825  
	https://blog.csdn.net/zsf442553199/article/details/79869377
	

==========How to freeze variable==========
	1. stop_gradient() & gradient()	
	2. get_variable/variable(trainable = False)
	3. 从trainOP上入手
	######### trainOp   variable_scope("discriminator")
	theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "discriminator")
	D_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(D_loss, var_list=theta_D)



	查看trainable变量
		for i in tf.trainable_variables():
			print(i.name)



"""


#### 1. tf.multiply   element-wise multiply
#### tf.matmul  matrix-multiply
'''
x=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]]) 
y=tf.constant([[0,0,1.0],[0,0,1.0],[0,0,1.0]])
#注意x,y要有相同的数据类型

x1=tf.constant(1)
y1=tf.constant(2)
z1=tf.multiply(x1,y1)

x2=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
y2=tf.constant(2.0)
z2=tf.multiply(x2,y2)

z=tf.multiply(x,y)
z3=tf.matmul(x,y)

with tf.Session() as sess:  
    print(sess.run([x,z]))
    print(sess.run(z1))
    print(sess.run(z2))
    print(sess.run(z3))
'''

############################################################################################################
#### 2. tf.cast
'''
a = tf.Variable([1,0,0,1,1])
b = tf.cast(a, dtype = tf.bool)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b))
'''

############################################################################################################
#### 3. tf.zeros / ones / zeros_like / ones_like 
'''
a = tf.Variable(tf.ones([2,3], dtype = tf.int32))
b = tf.Variable(tf.ones_like(a, dtype = tf.float32))
print('a name is {}'.format(a.name))
print('b name is {}'.format(b.name))
for i in tf.trainable_variables():
	print(i.name)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(([a,b])))
'''

############################################################################################################
#### 4. tf.constant & tf.Variable 
'''
a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.float64, name = 'Const1' )
a = tf.constant(-1.0, shape=[2, 3], dtype = tf.float64, name = 'Const1')
a = tf.constant([1,2,3,4,5,6], shape=[2, 3], dtype = tf.float64, name = 'Const1')

print(a.name)
with tf.Session() as sess:
	print(sess.run(a))

a = tf.Variable(tf.ones([2,3], dtype = tf.int32),  name = 'Variab1')	### trainable 
for i in tf.trainable_variables():
	print(i.name)
'''

############################################################################################################
#### 5. random_normal & random_uniform & random_shuffle 
'''
a = tf.random_normal([3,4,2], mean = 2, stddev = 1.0, dtype = tf.float32)
a = tf.random_uniform([3,4,2], minval = 0, maxval = 2, dtype = tf.float32)
va = tf.Variable(a, name = 'va')
vb = tf.random_shuffle(va)
vb = tf.Variable(vb, name = 'vb')
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(a))
for i in tf.trainable_variables():
	print(i.name)
'''


############################################################################################################
#### 6. tf.argmax & argmin
'''
#A = [[1,3,4,5,6]]
#B = [[1,3,4], [2,4,1]]
#A = tf.constant( [[1,3,4,5,6]] )
#C = tf.argmax(A,1)
A = tf.random_normal([3,4,2])
#A = np.random.random((3,4,2))		#### np.array !!!!
VA = tf.Variable(A)
C = tf.argmax(VA,1)
#C = tf.argmax(A,1)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#print(sess.run(tf.argmax(A, 1)))
	#print(sess.run(tf.argmax(B, 1)))
	print(sess.run([A, VA, C]))		### note that A and VA has different values. 
'''

############################################################################################################
### 7. tf.gather  tf.gather_nd *******   index gather? 
'''
A = tf.random_normal([3,4,2])

npA = np.random.random((3,4,2))
idx = [1,2]
#print(npA[idx])
#B = A[1,2,:]  # ok

with tf.Session() as sess:
 	print(sess.run(B))
'''
####  15.1 gather
'''a = tf.Variable([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
index_a = tf.Variable([0,2])
b = tf.Variable([1,2,3,4,5,6,7,8,9,10])
index_b = tf.Variable([2,4,6,8])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(tf.gather(a, index_a)))
	print(sess.run(tf.gather(b, index_b)))
'''
####   15.2 gather_nd **** https://qinqianshan.com/machine_learning/tensorflow/tf-gather_nd/
'''
a = tf.Variable([[1,2,3,4,5,6,7,8,9,10]])
y = tf.Variable([0,2])
p = tf.gather_nd(a,y) 
y_ = tf.Variable([[0,2]])
p_ = tf.gather_nd(a, y_)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([p, p_]))
'''
#############################################################
'''
indices = [[0, 0], [1, 1]]
params = [['a', 'b'], ['c', 'd']]
output1 = tf.gather_nd(params, indices)
indices = [[1], [0]]
params = [['a', 'b'], ['c', 'd']]
output2 = tf.gather_nd(params, indices)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([output1, output2]))
'''
###  indices最里面的括号代表要提取的params的坐标，信息提取后加入到括号。
'''
indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
indices_shape = tf.shape(indices)
params = [[['a0', 'b0'], ['c0', 'd0']],
          [['a1', 'b1'], ['c1', 'd1']]]
params_shape = tf.shape(params)
output = tf.gather_nd(params, indices)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([output, params_shape]))
'''
#############################################################
'''
#indices = [[[0, 0], [1, 0]], [[0, 1], [1, 1]], [[0, 1], [1, 1]]]  ### 3,2, (2)
#indices = [[[0, 0], [1, 0]]]  ### 1,2, (2)
indices = [[[0], [1]]]  ### 1,2, (1) (1)->2,4   => final shape is 1,2,2,4  
indices_shape = tf.shape(indices)
params = [[['a0', 'b0', ' ',  ' '], ['c0', 'd0', ' ',  ' ']],
          [['a1', 'b1', ' ',  ' '], ['c1', 'd1', ' ',  ' ']],
          [['a1', 'b1', ' ',  ' '], ['c1', 'd1', ' ',  ' ']]]  ### 3,2,4
params_shape = tf.shape(params)
output = tf.gather_nd(params, indices)
output_shape = tf.shape(output)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([output_shape, indices_shape, params_shape]))
'''



############################################################################################################
#### 8. tf.maximum & minimum;  greater & less & equal
'''
A = tf.random_normal([3,2])
B = tf.random_normal([3,2])
C = tf.maximum(A,B)
D = tf.maximum(A,0)
E = tf.less(A,B)
F = tf.less(A, 0.5)
G = tf.equal(A + 0.5, A + 0.5)
with tf.Session() as sess:
	#print(sess.run([A,B,C,D]))
	print(sess.run([A,B,E,F,G]))
'''

"""
A = tf.random_normal([3,2])
B = tf.maximum(A, 0)
with tf.Session() as sess:
	print(sess.run([A,B]))
""" 
############################################################################################################
#### 9. tf.add/div  log/exp
'''
A = tf.random_uniform([3,2], minval = 1, maxval = 2)
B = tf.random_uniform([3,2], minval = 1, maxval = 2)
C = tf.add(A,B)
D = tf.add(A,1)
C = tf.div(A,B)
D = tf.div(A,2)
C = tf.log(A)
D = tf.exp(C)

with tf.Session() as sess:
	print(sess.run([A,B,C,D]))
	#print(sess.run([A,B,E,F]))
'''

############################################################################################################
##############   shape 
#### 10. concat & stack & unstack  
##  tf.concat拼接的是除了拼接维度axis外其他维度的shape完全相同的张量，并且产生的张量的阶数不会发生变化，
##  而tf.stack则会在新的张量阶上拼接，产生的张量的阶数将会增加
'''
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
t1 = tf.Variable(t1, name = 't1')
t2 = tf.Variable(t2, name = 't2')
t3 = tf.concat([t1, t2], 0, name = 't3')
t3 = tf.Variable(t3, name = 't31')
t4 = tf.stack([t1, t2], axis = -1)  ### -1, 0, 1, 2 == -1
t4_shape = tf.shape(t4)
t5 = tf.unstack(t4, axis = -1)
t51 = t5[0]
t52 = t5[1]
t51bool = tf.equal(t51, t1)
t52bool = tf.equal(t52, t2)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(t3))
	print(sess.run([t4, t4_shape]))
	print(sess.run([t51bool, t52bool]))
for i in tf.trainable_variables():
	print(i.name)
'''
########################################
"""
t1 = tf.random_normal(shape = [32, 10, 1000], dtype = tf.float32)
t2 = tf.unstack(value = t1, num = 10, axis = 1)
assert isinstance(t2, list)
assert len(t2) == 10
assert t2[0].get_shape().as_list() == [32, 1000]
"""
'''
t1 = [tf.random_normal(shape = [30, 100], dtype = tf.float32) for i in range(5)]
t2 = tf.stack(values = t1, axis = 1)
assert t2.get_shape().as_list() == [30, 5, 100]
'''

############################################################################################################
####  11. tf.expand_dims & tf.squeeze
'''
t1 = tf.random_normal([2,3])
t2 = tf.expand_dims(t1, -2)  ### 0,1,2,-1, -2
t2_shape = tf.shape(t2)
t3 = tf.squeeze(t2, -2)
t3_shape = tf.shape(t3)
#t3 = tf.expand_dims(t1,-1)
#t3_shape = tf.shape(t3)

with tf.Session() as sess:
	print(sess.run([t2_shape, t3_shape]))
'''
############################################################################################################
####  12.  reshape & shape & get_shape(return a tuple, so can't use run) *******  & set_shape (different from reshape) & size(list, np.array, tf.tensor)
### dynamic shape , static shape   https://www.jianshu.com/p/2b88256ad206
#### get_shape(tensor)   tensor can be tf.ones(xxx); tf.variable(tf.random_uniform(xxx))
#### shape(list, array, tensor) use sess.run
'''
t1 = tf.random_normal([2,3,4])
t2 = tf.reshape(t1, [1,4,6])
t2_shape = tf.shape(t2)
#### size
t2_size = tf.size(t2)
x=tf.constant([[1,2,3],[4,5,6]])
x = tf.random_uniform([2,3], minval = 0, maxval = 1)
x = tf.Variable(x)
#### get_shape(can't use session run, because it return a tuple)
####  get_shape(tensor)   tensor can be tf.ones(xxx); tf.variable(tf.random_uniform(xxx));
x_get_shape = x.get_shape().as_list() 	### get_shape return a tuple. 
print(x.get_shape().as_list())

#### shape(list, array, tensor)
a = [[2,3], [0,1]]
list_shape = tf.shape(a)
a = np.zeros((3,2))
np_shape = tf.shape(a)

#### set_shape()
t1 = tf.placeholder(tf.int32)
t1.set_shape([3,2,4])
t1_shape = tf.shape(t1)

#t2_shape0 = tf.shape(t2)[0]
#t2_shape1 = tf.shape(t2)[1]
a = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
list_size = tf.size(a)
a = np.ones((2,3,4))
np_size = tf.size(a)

with tf.Session() as sess:
	print(sess.run([t2_shape, t2_size]))
	#print(sess.run([t2_shape0]))
	##print(sess.run(x_get_shape))	### can't use run ********
	print(sess.run([list_shape, np_shape]))
	print(sess.run(t1_shape))
	print(sess.run([list_size, np_size]))
'''
############################################################################################################


#### 13. transpose
'''
t1 = tf.random_normal([2,3,4])
t1_shape = tf.shape(t1)
t2 = tf.transpose(t1, perm = [2,1,0])
t2_shape = tf.shape(t2)

with tf.Session() as sess:
	print(sess.run([t1_shape, t2_shape]))
'''

############################################################################################################
#### 14. tf.tile   ## 堆叠 replicate
'''
a = tf.constant([[1,2],[3,4]],name='a') 
b = tf.tile(a,[2,3])
b_shape = tf.shape(b)

with tf.Session() as sess:
	print(sess.run(b_shape))
'''

############################################################################################################
#### 15. reduce_*: reduce_sum reduce_mean reduce_any
## reduce_all tf.reduce_max  reduce_logsumexp reduce_min  reduce_prod
'''
a = tf.random_normal([2,3,4])
sum_a = tf.reduce_sum(a, 1, keep_dims=False)
#### keep_dims=False => [2 4];  keep_dims=True => [2 1 4]
#sum_a = tf.reduce_mean(a,1)
sum_a_shape = tf.shape(sum_a)
with tf.Session() as sess:
	print(sess.run([sum_a_shape, a, sum_a]))
'''
############################################################################################################
#### 16. fill, only scalar 
####### contrast to tf.constant 
'''
a = tf.fill([2, 3], 9)
with tf.Session() as sess:
	print(sess.run(a))
'''
############################################################################################################
#### 17. where 
######### tf.gather_nd   where 和 gather_nd 互为逆过程

### usage 1
### example 1.
'''
input1 = [[[True, False], [True, False]], [[False, True], [False, True]], [[False, False], [False, True]]]   
#### 6,2
output1 = tf.where(input1)
input2 = tf.gather_nd(input1, output1)
with tf.Session() as sess:
	print(sess.run([output1, input2]))
'''
########################################################################
### example 2.
'''
input1 = tf.random_normal([4,3])
output1 = tf.where(tf.greater(input1, 0.5))
with tf.Session() as sess:
	print(sess.run([input1, output1]))
'''
############################################################################################

### usage 2
'''
x = [[1,2,3],[4,5,6]]
y = [[7,8,9],[10,11,12]]
condition3 = [[True,False,False],
             [False,True,True]]
with tf.Session() as sess:
	print(sess.run(tf.where(condition3,x,y)))
'''
############################################################################################################
#### 18. assign
### http://www.soaringroad.com/article/11?p=194  
### A must be Variable!!! 
'''
A = tf.Variable(tf.constant(0.0), dtype=tf.float32)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(5):
		sess.run(tf.assign(A, tf.add(A,10)))
		print(sess.run(A))
'''


############################################################################################################
#### 19. to_float  to_int  "cast" into float
'''
a = tf.constant([1,2,3])
print(a.dtype)
fa = tf.to_float(a)
print(fa.dtype)

a = tf.constant([1.2, 2.5, 3.9])
fa = tf.to_int32(a)
with tf.Session() as sess:
	print(sess.run(fa))
'''

############################################################################################################
#### 20. TensorShape is a class
'''
input0 = tf.constant([[0,1,2],[3,4,5]])

print(type(input0.shape))
print(type(input0.get_shape()))
print(type(tf.shape(input0)))

print(input0.shape.as_list())
print(input0.get_shape().as_list())
print(input0.shape.ndims)
print(input0.get_shape().ndims)
print(tf.rank(input0))
'''
#### 21. clip_by_value 
'''
A = tf.random_normal(shape = [3,4])
B = tf.clip_by_value(A, -1, 0.7)

with tf.Session() as sess:
	print(sess.run([A,B]))
'''


### 22. placeholder
'''
a = tf.placeholder(tf.float32, name = 'placeholder')
with tf.Session() as sess:
	print(sess.run([a], feed_dict = {a: 0.9}))
'''

### 23. tf.norm  matrix norm, normalized by rows/columns
"""
a = tf.random_normal(shape = [2,2], dtype = tf.float32)
b = tf.norm(tensor = a)	## F-norm 
a_l1 = tf.norm(tensor = a, ord = 1)
a_normalized = tf.nn.l2_normalize(a, axis = 1)  
### after normalize, each row vector has unit length.  
with tf.Session() as sess:
	print(sess.run([a,b, a_l1, a_normalized]))
"""


### 24. tf.nn.sigmoid_cross_entropy_with_logits 
'''
logits = tf.random_normal(shape = [2,2], dtype = tf.float32)
label = tf.ones_like(logits, dtype = tf.float32)

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
									labels = label)
loss_sum = tf.reduce_sum(loss, 1)
loss_mean = tf.reduce_mean(loss_sum)
with tf.Session() as sess:
	print( sess.run([logits, label, loss, loss_sum, loss_mean]) )
'''
### 25. tf.nn.softmax_cross_entropy_with_logits
'''
logits = tf.random_normal(shape = [2,2], dtype = tf.float32)
labels = tf.placeholder(dtype = tf.int32, shape = [None, 2])
lab = [[1,0], [0,1]]
loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
with tf.Session() as sess:
	print(sess.run([logits, labels, loss], feed_dict = {labels: lab}))
'''
"""
[array([[-1.5072867 , -0.8529847 ],
       [-0.78440124, -0.26547712]], dtype=float32), array([[1, 0],
       [0, 1]], dtype=int32), array([1.0728838 , 0.46697435], dtype=float32)]

exp(-1.5072867) / (exp(-1.5072867) + exp(-0.8529847))   =>  0.34202074733030896
- log(0.34202074733030896) => 1.072883879051055
"""


###  26. embedding_lookup
'''
embed_mat = np.random.random([5,2])
seq_idx = tf.placeholder(dtype = tf.int32, shape = [None, 3])
lst = [[1,2,3], [0,1,4]]
#lst = [[1,2,3], [0,1,4]]

embedded_seq = tf.nn.embedding_lookup(params = embed_mat, ids = seq_idx)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	embedded_seq0 = sess.run((embedded_seq), 
								feed_dict = {seq_idx: lst})
print(embed_mat)
print(embedded_seq0)
print(type(embedded_seq0))
#assert embedded_seq0.get_shape().as_list() == [2, 3, 2]
assert embedded_seq0.shape == (2,3,2)
'''

"""
###### Interesting problem
c = np.random.random([10,1])
b = tf.nn.embedding_lookup(c, [1, 3])
 
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(c))
### why is it a bug?   It is interesting ********************************** Answer is sess.run's target object can only be tensor, not list, not np.array.
"""


### 27. tf.nn.conv1d & tf.layers.conv1d
'''
embed_dim = 15
length = 2
conv_kernel_size = 11
out_filter = 9
pool_size = 6
inputs = tf.placeholder('float', shape=[None, length, embed_dim])   
#### length is 6; embedding dimension is 8; 
out = tf.layers.conv1d(inputs = inputs, filters = out_filter, kernel_size = conv_kernel_size, strides = 1, padding = 'same')	
### padding:  'valid', 'same'. 

mp = tf.layers.max_pooling1d(inputs = out, pool_size = pool_size, strides = 1, padding = 'same')

np_inputs = np.random.random((10, length, embed_dim))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#conv, = sess.run([out], feed_dict = {inputs: np_inputs})
	#print(conv.shape)
	conv, mpool  = sess.run([out, mp], feed_dict = {inputs: np_inputs})
	print(mpool.shape)
	#assert mpool.shape[1] == length - conv_kernel_size + 1 - pool_size + 1
'''

"""
input:  batch_size = 10, length = 6; embed_dim = 8
filter num = 5, kernerl = 3, stride = 1, padding = 'valid'
output: batch_size = 10, new_length = 5, out_channel = 9
10, 5, 9
10, 3, 9
"""

### 28. tf.Tensor => np.array, no direct way
'''
z = tf.random_normal([2, 3])
with tf.Session() as sess:
	z_np = z.eval()
print(z_np)
'''


### 29. tf dtype
a = tf.random_normal([2,3],dtype = tf.float32)
assert a.dtype == tf.float32













