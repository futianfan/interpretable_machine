import tensorflow as tf
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(3)

""" 
1. tf.matrix, tf.matmul, 
2. tf.cast 
3. tf.zeros / ones / zeros_like / ones_like 
4. tf.constant & tf.Variable 
5. random_normal & random_uniform & random_shuffle 
6. tf.argmax  argmin 
7. index gather? 
8. tf.maximum & minimum;  greater & less & equal
9. tf.add/div  log/exp
10. concat
### list of list; numpy.array can be used as input for TF 
"""


#### 1. tf.matrix   element-wise multiply
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


#### 2. tf.cast
'''
a = tf.Variable([1,0,0,1,1])
b = tf.cast(a, dtype = tf.bool)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b))
'''


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


### 7. index gather? 
'''
A = tf.random_normal([3,4,2])

npA = np.random.random((3,4,2))
idx = [1,2]
#print(npA[idx])
#B = A[1,2,:]  # ok

with tf.Session() as sess:
 	print(sess.run(B))
'''


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


##############   shape 
#### 10. concat  
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
t1 = tf.Variable(t1, name = 't1')
t2 = tf.Variable(t2, name = 't2')
t3 = tf.concat([t1, t2], 0, name = 't3')
t3 = tf.Variable(t3, name = 't31')
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(t3))

for i in tf.trainable_variables():
	print(i.name)



