import numpy as np 
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc

__all__ = [
	'GAN',
]

class GAN(object):

	def __init__(self, sess, FLAGS):

		self.img_dim = FLAGS.img_dim
		self.z_dim = FLAGS.z_dim
		self.g_hidden_size = FLAGS.g_hidden_size 
		self.d_hidden_size = FLAGS.d_hidden_size
		self.learning_rate = FLAGS.learning_rate
		self.build_model()
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())		

	def build_model(self):
		### tf.placeholder
		self.x = tf.placeholder(tf.float32, shape = [None, self.img_dim])
		self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim])

		######### generator forward 
		G_logit, G = self.generator(self.z)
		######### discriminator forward
		D_logit, D = self.discriminator(self.x, reuse = tf.AUTO_REUSE)	### real
		D_logit_fake, D_fake = self.discriminator(G, reuse = tf.AUTO_REUSE)  ### fake

		######### loss
		self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit, labels=tf.ones_like(D_logit)))
		self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
		self.D_loss = self.D_loss_real + self.D_loss_fake
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.ones_like(D_logit_fake)))

		for i in tf.trainable_variables():
			print(i.name)
		self.theta_D = [i for i in tf.trainable_variables() if 'dis_' in i.name]
		self.theta_G = [i for i in tf.trainable_variables() if 'gen_' in i.name]
		######### trainOp
		#self.theta_D2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                             "discriminator")
		#self.theta_G2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                             "generator")
		#assert self.theta_D == self.theta_D2 and self.theta_G == self.theta_G2
		self.D_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.D_loss, var_list=self.theta_D)
		self.G_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.G_loss, var_list=self.theta_G)  ### , var_list = self.theta_G
		#return 

	def discriminator(self, x, reuse = False):
		with tf.variable_scope('discriminator', reuse = reuse) as scope:  ### reuse = tf.AUTO_REUSE
			#if reuse:
			#	scope.reuse_variables() 
			'''
			w1 = tf.Variable(tf.random_normal(shape = [x.get_shape()[1], self.d_hidden_size], dtype = tf.float32))
			b1 = tf.Variable(tf.zeros([self.d_hidden_size],dtype=tf.float32))
			w2 = tf.Variable(tf.random_normal(shape = [self.d_hidden_size, 1], dtype = tf.float32))
			b2 = tf.Variable(tf.zeros([1],dtype=tf.float32))
			h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
			h2 = tf.matmul(h1,w2) + b2
			h2_act = tf.nn.sigmoid(h2)'''			
			d1 = fc(x, self.d_hidden_size, scope = 'dis_fc1', activation_fn = tf.nn.relu)
			d_log = fc(d1, 1, scope = 'dis_fc2', activation_fn = None)
			d2 = tf.nn.sigmoid(d_log)
		return d_log, d2 
		#return h2, h2_act

	'''def generator(self, z, reuse = False):
		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables()
			g1 = fc(self.z, self.g_hidden_size, scope = 'gen_fc1', activation_fn = tf.nn.relu)
			g_log = fc(g1, self.img_dim, scope = 'gen_fc2', activation_fn = None)
			g2 = tf.nn.sigmoid(g_log)
		return g_log, g2 '''
	def generator(self, z, reuse = False):
		with tf.variable_scope('generator') as scope:
			if reuse:
				scope.reuse_variables()
			w1 = tf.Variable(tf.random_normal(shape=[]))
			g1 = fc(z, self.g_hidden_size, scope = 'gen_fc1', activation_fn = tf.nn.relu)
			g_log = fc(g1, self.img_dim, scope = 'gen_fc2', activation_fn = None)
			g2 = tf.nn.sigmoid(g_log)
		return g_log, g2 


	
	def sample_z(self, batch_size):
		return np.random.uniform(-1, 1, size = [batch_size, self.z_dim])

	def train(self, x):
		batch_size = x.shape[0]
		sampled_z = self.sample_z(batch_size)
		_, dis_loss = self.sess.run((self.D_solver, self.D_loss), 
			feed_dict = {self.x:x, self.z:sampled_z})
		_, gen_loss = self.sess.run((self.G_solver, self.G_loss), 
			feed_dict = {self.z:sampled_z})
		return dis_loss, gen_loss
		#return dis_loss, 0

if __name__ == '__main__':
	pass 


"""
TO DO LIST

1.		self.G_solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.G_loss)  ### , var_list = self.theta_G

2.		_, gen_loss = self.sess.run((self.G_solver, self.G_loss), 
			feed_dict = {self.z:sampled_z})

"""



















