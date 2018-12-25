import tensorflow as tf
import numpy as np

__all__ = [
	"autoencoders",
	"decoder"
]


def _gaussian_mlp_encoder(x, hidden_size1, hidden_size2, output_dim, dropout_keep_prob, lower_bound_of_stddev, reuse):
	#with tf.variable_scope('gaussian_encoder', reuse = reuse):
	with tf.variable_scope('gaussian_encoder', reuse = tf.AUTO_REUSE):
	#with tf.variable_scope('gaussian_encoder'):
		## initialize
		weight_init = tf.contrib.layers.variance_scaling_initializer()
		bias_init = tf.constant_initializer(0.)

		## 1st layer
		w0 = tf.get_variable('w0', shape = [x.get_shape()[1], hidden_size1], initializer = weight_init)
		b0 = tf.get_variable('b0', shape = [hidden_size1], initializer = bias_init)
		h0 = tf.matmul(x, w0) + b0
		h0 = tf.nn.elu(h0)
		h0 = tf.nn.dropout(h0, dropout_keep_prob)

		## 2nd layer
		w1 = tf.get_variable('w1', shape = [h0.get_shape()[1], hidden_size2], initializer = weight_init)
		b1 = tf.get_variable('b1', shape = [hidden_size2], initializer = bias_init)
		h1 = tf.matmul(h0, w1) + b1
		h1 = tf.nn.tanh(h1)
		h1 = tf.nn.dropout(h1, dropout_keep_prob)

		## gaussian params
		wout = tf.get_variable('wout', shape = [h1.get_shape()[1], 2 * output_dim], initializer = weight_init)
		bout = tf.get_variable('bout', shape = [output_dim * 2], initializer = bias_init)
		gauss_params = tf.add(tf.matmul(h1, wout),  bout )
		mean = gauss_params[:,:output_dim]
		stddev = lower_bound_of_stddev + tf.nn.softplus(gauss_params[:,output_dim:])
	return mean, stddev

def _bernoulli_decoder(z, hidden_size1, hidden_size2, output_dim, dropout_keep_prob, reuse):
	#with tf.variable_scope('bernoulli_decoder', reuse = reuse):
	with tf.variable_scope('bernoulli_decoder', reuse = tf.AUTO_REUSE):
		weight_init = tf.contrib.layers.variance_scaling_initializer()
		bias_init = tf.constant_initializer(0.)

		## 1st layer 
		w0 = tf.get_variable('w0', shape = [z.get_shape()[1], hidden_size1], initializer = weight_init)
		b0 = tf.get_variable('b0', shape = [hidden_size1], initializer = bias_init)
		h0 = tf.matmul(z, w0) + b0
		h0 = tf.nn.tanh(h0)
		h0 = tf.nn.dropout(h0, dropout_keep_prob)

		## 2nd layer 
		w1 = tf.get_variable('w1', shape = [h0.get_shape()[1], hidden_size2], initializer = weight_init)
		b1 = tf.get_variable('b1', shape = [hidden_size2], initializer = bias_init)
		h1 = tf.matmul(h0, w1) + b1 
		h1 = tf.nn.elu(h1)
		h1 = tf.nn.dropout(h1, dropout_keep_prob)

		## output layer 
		wout = tf.get_variable('wout', shape = [h1.get_shape()[1], output_dim], initializer = weight_init)
		bout = tf.get_variable('bout', shape = [output_dim], initializer = bias_init)
		y = tf.nn.sigmoid(tf.matmul(h1, wout) + bout)
	return y 


def autoencoders(x_in, x_trg, img_dim, z_dim, hidden_size1, hidden_size2, dropout_keep_prob, lower_bound_of_stddev, clip_threshold):

	## encoder
	mu, sigma = _gaussian_mlp_encoder(x_in, hidden_size1, hidden_size2, z_dim, dropout_keep_prob, lower_bound_of_stddev, reuse = False)



	## sample z
	z = mu + sigma * tf.random_normal(shape = tf.shape(mu), mean = 0.0, stddev = 1.0, dtype = tf.float32)

	## decoder 
	y = _bernoulli_decoder(z, hidden_size2, hidden_size1, img_dim, dropout_keep_prob, reuse = False)
	y = tf.clip_by_value(y, clip_threshold, 1 - clip_threshold)

	## loss
	marginal_likelihood = tf.reduce_sum(x_trg * tf.log(y) + (1 - x_trg) * tf.log(1 - y), 1)
	KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

	marginal_likelihood = tf.reduce_mean(marginal_likelihood)
	KL_divergence = tf.reduce_mean(KL_divergence)
	ELBO = marginal_likelihood - KL_divergence
	loss = -ELBO

	return y, z, loss, -marginal_likelihood, KL_divergence

def decoder(z, hidden_size1, hidden_size2, output_dim):
	return _bernoulli_decoder(z, hidden_size1, hidden_size2, output_dim, dropout_keep_prob = 1.0, reuse = True)








