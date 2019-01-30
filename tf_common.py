import tensorflow as tf
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(3)
np.random.seed(1)
tf.set_random_seed(5)

""" 

tf.reverse
tf.scan
tf.map_fn
tf.convert_to_tensor ***
tf.stop_gradient & tf.gradient 

tf.identity:   Return a tensor with the same shape and contents as input.
tf.range: range, np.arange()   tf.range(start, limit, delta)
tf.cond: if ... else ...

tf.equal 
tf.cast
tf.tensordot 
tf.while_loop 

==================================== NLP term ====================================
tf.one_hot  
tf.pad
tf.sequence_mask
==================================== NLP term ====================================



tf.to_int32 / to_int64
tf.to_float ***

tf.name_scope
tf.get_variable(name=hparams["name"],
tf.Variable

get_collection
get_collection_ref

tf.local_variables_initializer
tf.tables_initializer


tf.TensorShape	******* see 20.TensorShape:   tf.shape  get_shape


tf.contrib.layers.fully_connected
tf.contrib.rnn.RNNCell
tf.contrib.seq2seq.SampleEmbeddingHelper
tf.contrib.seq2seq.TrainingHelper(
tf.contrib.seq2seq.BasicDecoder(
f.contrib.seq2seq.dynamic_decode(tf_decoder)
classtype=tf.contrib.seq2seq.AttentionMechanism)
tf.contrib.seq2seq.tile_batch(initial_state, inputs_shape[2])

tf.distributions.Categorical(logits=logits)

tf.logical_and / logical_not


Modules

	tf.layers   93
	tf.nn 		57
	tf.contrib   19 
	tf.contrib.seq2seq 17 
	tf.estimator 99
	tf.train 	23	
	tf.app
	tf.gfile
	tf.test.TestCase*** 
	tf.data******
	tf.compat***** 




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
30 tf.scatter_update scatter_nd ------indexing & update 
31 tf.exp / div / sqrt / rsqrt (1 / sqrt(x))
32 tf.compat.as_str / as_bytes / as_text
33 tf.rank:  Equivalent to np.ndim
34 tf.split:  
35 tf.tensordot
36 tf.nn.softmax 
37 eval
38  embedding + embedding_lookup
39 LSTM: bidirectional + multilayer 
### list of list; numpy.array can be used as input for TF 





grep "tf\." `find . -name \*.py` |  grep -v "tf.matmul" | grep -v "tf.multiply"\
 | grep -v tf.ones | grep -v tf.zeros | grep -v tf.random_normal | \
 grep -v tf.random_uniform \
 | grep -v tf.placeholder \
 | grep -v tf.embedding_lookup \
 | grep -v tf.clip_by_value \
 | grep -v tf.stack \
 | grep -v tf.unstack \
 | grep -v tf.concat \
 | grep -v tf.ones_like \
 | grep -v tf.zeros_like \
 | grep -v tf.gather \
 | grep -v tf.maximum \
 | grep -v tf.minimum \
 | grep -v tf.transpose \
 | grep -v tf.reduce_ \
 | grep -v tf.norm \
 | grep -v tf.layers\
 | grep -v tf.nn\
 | grep -v tf.estimator\
 | grep -v tf.train\
 | grep -v tf.int \
 | grep -v tf.float \
 | grep -v variable_scope \
 | grep -v tf.reshape\
 | grep -v tf.expand_dims \
 | grep -v tf.squeeze \
 | grep -v tf.constant\
 | grep -v tf.tile \
 | grep -v tf.floor \
 | grep -v tf.test.TestCase \
 | grep -v tf.test.main \
 | grep -v tf.global_variables_initializer \
 | grep -v tf.add \
 | grep -v tf.div \
 | grep -v tf.gfile \
 | grep -v tf.contrib \
 | grep -v tf.app \
 | grep -v tf.distribution \
 | grep -v tf.stop_gradient \
 | grep -v tf.equal \
 | grep -v tf.one_hot \
 | grep -v tf.scatter \
 | grep -v tf.shape \
 | grep -v tf.compat.as_text \
 | grep -v tf.logical_ \
 | grep -v tf.to_float \
 | grep -v tf.to_int \
 | grep -v tf.sequence_mask \
 | grep -v tf.TensorShape \
 | grep -v tf.where \
 | grep -v tf.identity \
 | grep -v tf.string \
 | grep -v tf.abs \
 | grep -v tf.name_scope \
 | grep -v tf.get_variable \
 | grep -v tf.bool \
 | grep -v tf.split \
 | grep -v tf.argmax \
 | grep -v tf.greater \
 | grep -v tf.less \
 | grep -v tf.pad \
 | grep -v tf.rank \
 | grep -v tf.compat.as_bytes \
 | grep -v tf.fill \
 | grep -v tf.cast \
 | grep -v tf.exp \
 | grep -v tf.cond \
 | grep -v tf.convert_to_tensor\
 | grep -v tf.cum \
 | grep -v tf.log \
 | grep -v tf.tables_initializer \
 | grep -v tf.local_variables_initializer \
 | grep -v tf.errors \
 | grep -v tf.get_collection \
 | grep -v tf.Session \
 | grep -v tf.clip_by_global_norm \
 | grep -v tf.assign \
 | grep -v tf.make_template \
 | grep -v tf.get_collection_ref \
 | grep -v tf.keras.regularizers \
 | grep -v tf.data \
 | grep -v tf.map_fn \
 | grep -v tf.compat.as_str \
 | grep -v tf.size \
 | grep -v tf.while_loop \
 | awk -F ":" '{print $2}' | grep tf


grep tf.errors `find . -name \*.py` | wc -l






advanced topic
	Variable  get_variable
	variable_scope
	## Tensorflow踩坑记  https://blog.csdn.net/hejunqing14/article/details/52688825  
	https://blog.csdn.net/zsf442553199/article/details/79869377
	
	tf.TensorShape  tf.shape   get_shape

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

#### 补充 from 12.1 tf.size: input can be list, np.array, tensor 
'''
a = [[1,2], [3,4]]
a = np.random.random((2,3))
size_a = tf.size(a)
with tf.Session() as sess:
	print(sess.run(size_a))
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
#### 20. TensorShape**** is a class
### get_shape, tf.shape()  get_shape returns a tuple. 

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
"""
a = tf.random_normal([2,3],dtype = tf.float32)
assert a.dtype == tf.float32
"""


###  example 1: compute weighted cross-entropy loss. 
'''
logits = tf.random_normal([3,2], dtype = tf.float32)
y = tf.placeholder(shape = [None, 2], dtype = tf.int32)
Tweight = tf.placeholder(shape = [None], dtype = tf.float32)
list_y = [[1, 0], [0, 1], [1, 0]]
weight = [1, 2, 3]
loss = tf.nn.softmax_cross_entropy_with_logits(
										labels=y, 
										logits=logits)
weighted_loss = tf.reduce_mean(Tweight * loss)
with tf.Session() as sess:
	print(sess.run([loss, weighted_loss], feed_dict={y: list_y, Tweight:weight}))
'''


### example 2: feedforward prototype layer 

"""
tf.shape 必须通过sess.run;   X.get_shape() 构图的时候出现None， 构图的时候需要batch_size这个信息  如何处理？
"""

"""
batch_size = 2
hidden_dim = 2
prototype_num = 3
X_2 = tf.random_normal([batch_size, hidden_dim])
X_4 = np.random.random((batch_size, hidden_dim))

### build graph 
X_ = tf.placeholder(shape = [None, hidden_dim], dtype = tf.float32)
X = tf.placeholder(shape = [batch_size, hidden_dim], dtype = tf.float32)
prototype_vector_ = tf.random_normal([prototype_num, hidden_dim])
'''
def prototype_layer(X, prototype_vector):
	#with tf.variable_scope('prototype') as scope:
	#batch_size, hidden_dim = tf.shape(X)
	#prototype_num, hidden_dim = tf.shape(prototype_vector)
	batch_size, hidden_dim = X.get_shape()
	prototype_num, hidden_dim = prototype_vector.get_shape()	
	X_extend = tf.stack([X] * prototype_num, axis = 1)
	prototype_vector_extend = tf.stack([prototype_vector] * batch_size, axis = 0)
	X_dif = X_extend - prototype_vector_extend
	output = tf.norm(X_dif, ord = 'euclidean', axis = 2)
	return output
'''
#output = prototype_layer(X_, prototype_vector_)
'''
batch_size, hidden_dim = tf.shape(X_)
prototype_num, hidden_dim = tf.shape(prototype_vector_)
X_extend = tf.stack([X] * prototype_num, axis = 1)
prototype_vector_extend = tf.stack([prototype_vector] * batch_size, axis = 0)
X_dif = X_extend - prototype_vector_extend
output = tf.norm(X_dif, ord = 'euclidean', axis = 2)
'''
def prototype_layer(X, prototype_vector):
	### X: b, d
	### prototype_vector:   p, d
	prototype_num, hidden_dim = prototype_vector.get_shape()
	X_extend = tf.stack([X] * prototype_num, axis = 1)	#### b, p, d
	X_extend_split = tf.unstack(value = X_extend, axis = 0)  ### list of length b, each element is p,d
	X_2 = [tf.norm(X_i - prototype_vector, ord = 'euclidean', axis = 1)   \
								for X_i in X_extend_split]
	X_3 = tf.stack(X_2, axis = 0)
	#assert X_3.get_shape() == (batch_size, prototype_num)
	return X_3

#output = prototype_layer(X, prototype_vector_)
output = prototype_layer(X_, prototype_vector_)
"""

'''
with tf.Session() as sess:
	#a, b = sess.run([X_extend, prototype_vector_extend])
	#assert a.shape == (batch_size, prototype_num, hidden_dim)
	#assert b.shape == (batch_size, prototype_num, hidden_dim)
	#c = sess.run([output], feed_dict = {X_:X_2})
	c = sess.run([output], feed_dict = {X:X_4})	
	c = c[0]
	#print(c.shape)
	assert c.shape == (batch_size, prototype_num)
	#print(sess.run([X_, prototype_vector_, output]))
'''


### example 2.1 
'''
batch_size = 2
hidden_dim = 2
prototype_num = 3
npx = np.random.random((batch_size, hidden_dim))
X_ = tf.placeholder(shape = [None, hidden_dim], dtype = tf.float32)
y = tf.random_normal([hidden_dim])
#X = tf.unstack(value = X_,  axis = 0)
#print(X_.get_shape()[0])
X = X_ - y
with tf.Session() as sess:
	print(sess.run([X], feed_dict = {X_:npx}))
'''


### example 2.2
'''
batch_size = 4
hidden_dim = 2
prototype_num = 3
npx = np.random.random((batch_size, hidden_dim))
X_ = tf.placeholder(shape = [None, hidden_dim], dtype = tf.float32)
prototype_vector_ = tf.random_normal(shape = [prototype_num, hidden_dim])


def prototype_layer(X_, prototype_vector_):
	"""
		X_: b, d   batch_size  , dim   ***** b is None owing to ***placeholder****
		prototype_vector_: p, d   prototype_num, dim
	"""
	prototype_num = prototype_vector_.get_shape()[0]
	for i in range(prototype_num):
		y = X_ - tf.gather(prototype_vector_, [i])
		#y = X_ - tf.gather(prototype_vector_, tf.Variable([i]))		
		y = tf.norm(y, ord = 'euclidean', axis = 1)
		y = tf.reshape(y, [1, -1])
		if i == 0:
			output = y 
		else:
			output = tf.concat([output, y], 0)
	output = tf.transpose(output, perm = [1,0])  ### p,b => b,p
	return output

output = prototype_layer(X_, prototype_vector_)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([output], feed_dict = {X_:npx}))
'''



### example 2.3
'''
batch_size = 4
hidden_dim = 2
prototype_num = 3
npx = np.random.random((batch_size, hidden_dim))
X_ = tf.placeholder(shape = [None, hidden_dim], dtype = tf.float32)
prototype_vector_ = tf.random_normal(shape = [prototype_num, hidden_dim])


def prototype_layer(X_, prototype_vector_):
	"""
		X_: b, d   batch_size  , dim   ***** b is None owing to ***placeholder****
		prototype_vector_: p, d   prototype_num, dim
	"""
	prototype_num = prototype_vector_.get_shape()[0]
	X_extend = tf.stack([X_] * prototype_num, axis = 1)
	assert X_extend.get_shape()[1] == prototype_num
	prototype_vector_extend = tf.expand_dims(prototype_vector_, 0)
	assert prototype_vector_extend.get_shape()[:2] == (1, prototype_num)
	return tf.norm(X_extend - prototype_vector_extend, axis = 2)


output = prototype_layer(X_, prototype_vector_)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([output], feed_dict = {X_:npx}))

'''



### example 2.4: return minimum euclidean distance for prototype loss
'''
batch_size = 4
hidden_dim = 2
prototype_num = 3
#npx = np.random.random((batch_size, hidden_dim))
npx = np.zeros((batch_size, hidden_dim))
X_ = tf.placeholder(shape = [None, hidden_dim], dtype = tf.float32)
prototype_vector_ = tf.random_normal(shape = [prototype_num, hidden_dim])


def prototype_loss(X_, prototype_vector_):
	"""
		X_: ?, d   batch_size  , dim   ***** ?==b is None owing to ***placeholder****
		prototype_vector_: p, d   prototype_num, dim
	"""
	prototype_num = prototype_vector_.get_shape()[0]
	X_extend = tf.stack([X_] * prototype_num, axis = 1)  ### ?, p, d
	prototype_vector_extend = tf.expand_dims(prototype_vector_, 0)  ### 1, p, d
	X_dif = tf.norm(X_extend - prototype_vector_extend, ord = 'euclidean', axis = 2)  ### ?, p
	X_min = tf.reduce_min(X_dif, axis = 1)
	return X_min


output = prototype_loss(X_, prototype_vector_)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([X_, prototype_vector_, output], feed_dict = {X_:npx}))

'''

#### 30 tf.scatter_update scatter_nd ------indexing & update 
###  tf.gather tf.gather_nd
###  a is tf.Tensor	a[1] = blabla

#######  scatter_update
'''
a = tf.Variable(tf.random_normal([5,4]))
b = tf.scatter_update(a, [0, 1], [[1, 1, 0, 0], [1, 0, 4, 0]])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	c = sess.run(b)
	print(type(c))

	#print(sess.run([b]))
'''

###### scatter_nd   TO DO 


###### scatter_add 


#### 31 tf.exp / div / sqrt / rsqrt (1 / sqrt(x))
'''
a = tf.random_uniform(shape = [3,4], minval = 0, maxval = 1, dtype = tf.float32)
b = tf.random_uniform(shape = [3,1], minval = 0, maxval = 1, dtype = tf.float32)
sqrt_a = tf.rsqrt(a)
div_ab = tf.div(a,b)

with tf.Session() as sess:
	print(sess.run([a, b, sqrt_a, div_ab]))
'''


##### 32 tf.compat.as_str / as_bytes / as_text 
####??? how to use, encode
####  Functions for Python 2 vs. 3 compatibility.


'''
lines = ['abc c fewj \n', 'bbc feo fe \n', 'aaai']

line = tf.compat.as_bytes(lines[0])
line2 = tf.compat.as_str(lines[1])
line3 = tf.compat.as_text(lines[1])
print(line)
print(line2)
print(line3)
#### don't have sess.run
'''


#### 33 tf.rank:  Equivalent to np.ndim
'''
a = tf.random_normal(shape = [1,2,3,4,5])
rank_a = tf.rank(a)
with tf.Session() as sess:
	print(sess.run(rank_a))

b = np.random.random((1,2,3))
print(b.ndim)
'''



#### 34 tf.split
## example 1.
'''
a = tf.random_uniform(shape = [2,8], minval = 0, maxval = 1, dtype = tf.float32)
b = tf.split(value = a, num_or_size_splits = 4, axis = 1)
#### 8 被 4 整除 
with tf.Session() as sess:
	print(sess.run([a,b]))
'''


## example 2.
# axis为1，所以value.shape[1]为30，4+15+11正好为30
'''
value = tf.random_normal([2,30])
split0, split1, split2 = tf.split(value = value, num_or_size_splits = [4, 15, 11], axis = 1)

a = tf.shape(split0)  # [5, 4]
b = tf.shape(split1)  # [5, 15]
c = tf.shape(split2)  # [5, 11]

with tf.Session() as sess:
	print(sess.run([value, a,b,c]))
'''

#### 35 tf.tensordot: similar to np.tensordot , details see np_common.py 
'''
a = tf.ones([5,4,2,3])
b = tf.ones([2,3,6])
shape1 = tf.tensordot(a, b, axes = 2).get_shape().as_list()
assert shape1 == [5, 4, 6]
'''

'''
a = tf.ones([5,4,2,3])
b = tf.ones([3,2,6])
shape1 = tf.tensordot(a,b, axes = (2,1)).get_shape().as_list()
assert shape1 == [5,4,3,3,6]

shape2 = tf.tensordot(a, b, axes = (3, 0)).get_shape().as_list()
assert shape2 == [5, 4, 2, 2, 6]

shape3 = tf.tensordot(a,b, axes = ((2, 3), (1, 0))).get_shape().as_list()
assert shape3 == [5, 4, 6]

shape4 = tf.tensordot(a,b, axes = ((-2, -1), (1, 0)) ).get_shape().as_list()
assert shape4 == [5,4,6]
'''




### 36 tf.nn.softmax   默认 B,D i.e., axis = -1, 最后一个维度  
'''
a = tf.random_normal([2,3])
b = tf.nn.softmax(a)

with tf.Session() as sess:
	print(sess.run([a,b]))
'''
'''
a = tf.random_normal([2,2,2])
b = tf.nn.softmax(a)
c = tf.reduce_sum(b, axis = -1)
with tf.Session() as sess:
	print(sess.run([a,b,c]))
'''



### example: attention mechanism
"""
attention_weight (alpha): B, T
input: B, T, D
output: B, D
"""
'''
B, T, D = 2, 3, 2
X_in = tf.ones(shape = [B, T, D])
alpha = tf.random_uniform(shape = [B, T], minval = 0, maxval = 1)
alpha_expand = tf.expand_dims(alpha, -1)  ### B, T, 1
X_o = tf.multiply(X_in, alpha_expand)

with tf.Session() as sess:
	print(sess.run([X_in, alpha, X_o]))

'''

### 37 eval()   get value from tf
'''
a = tf.Variable(tf.random_normal([3,4]))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	b = a.eval()
	print(b)
'''


###  38  embedding + embedding_lookup & LSTM: bidirectional + multilayer
##  https://github.com/dongjun-Lee/birnn-language-model-tf/blob/master/model/bi_rnn_lm.py
##  https://stackoverflow.com/questions/46011973/attributeerror-lstmstatetuple-object-has-no-attribute-get-shape-while-build
'''
from tensorflow.contrib import rnn
word_num = 10
embed_dim = 5
rnn_hidden_size = 36
keep_prob = 0.8
rnn_num_layer = 2
X = [[1,2,3,1], [1,2,1,0], [0,0,2,1]]  ### 3, 4	
src_in = tf.placeholder(shape = [None, 4], dtype = tf.int32)

### embedding 
with tf.variable_scope('embedding'):
	embedding_mat = tf.get_variable(
							'lookup',
							shape = [word_num, embed_dim],
							initializer=tf.random_uniform_initializer(minval = -1, maxval = 1), 
							trainable = True
						)
### embedding lookup
src_embed = tf.nn.embedding_lookup(embedding_mat, src_in)

### bi-lstm multiple layer
with tf.variable_scope('bi-rnn'):
	def make_cell():
		cell = rnn.BasicLSTMCell(rnn_hidden_size)
		cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
		return cell 

	fw_cell = rnn.MultiRNNCell([make_cell() for _ in range(rnn_num_layer)])
	bw_cell = rnn.MultiRNNCell([make_cell() for _ in range(rnn_num_layer)])
	((encoder_fw_outputs, encoder_bw_outputs),
	 (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
		fw_cell, bw_cell, src_embed, dtype = tf.float32)


	encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), axis = 2)
	encoder_states = []

	for i in range(rnn_num_layer):
		if isinstance(encoder_fw_state[i],tf.contrib.rnn.LSTMStateTuple):
			print('ok')
			encoder_state_c = tf.concat(values=(encoder_fw_state[i].c,encoder_bw_state[i].c),axis=1,name="encoder_fw_state_c")
			encoder_state_h = tf.concat(values=(encoder_fw_state[i].h,encoder_bw_state[i].h),axis=1,name="encoder_fw_state_h")
			encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
		elif isinstance(encoder_fw_state[i], tf.Tensor):
			encoder_state = tf.concat(values=(encoder_fw_state[i], encoder_bw_state[i]), axis=1, name='bidirectional_concat')

		encoder_states.append(encoder_state)
'''

#### conv2d 
"""
from tensorflow.python.ops import nn_ops
#### B,T,1,D
B,T,D = 3,4,2
encoder_states = tf.Variable(tf.random_normal(shape = [B,T,1,D]))
W_h = tf.Variable(tf.random_normal(shape = [1,1,D,D]))
encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	_, _, out =  sess.run([encoder_states, W_h, encoder_features])
	print(out.shape)
"""

'''
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	src_embed_arr =  sess.run([src_embed], feed_dict = {src_in: X})	
	src_embed_arr = src_embed_arr[0]
	print(src_embed_arr.shape)
'''



### mask
def mask_normalize_attention(padding_mask, attention_weight):
	"""
		padding_mask: B,T
		attention_weight: B,T
	"""
	padding_mask = tf.cast(padding_mask, dtype = tf.float32)
	attention_weight *= padding_mask
	attention_weight_sum = tf.reduce_sum(attention_weight, 1)
	return attention_weight / tf.reshape(attention_weight_sum, [-1,1])

weight = tf.Variable(tf.random_uniform(shape = [3,4], minval = 0, maxval = 1, dtype = tf.float32))
padding_mask = tf.placeholder(tf.int32, shape = [3,4])

weight_mask = mask_normalize_attention(padding_mask, weight)

padding = [[1,1,1,0], [1,1,0,0], [1,0,0,0]]


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run([weight, weight_mask], feed_dict = {padding_mask:padding}))



