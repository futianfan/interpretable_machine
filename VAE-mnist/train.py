import tensorflow as tf
import numpy as np



def train():
	import stream
	from config import get_config
	import model
	config = get_config()
	## parameter 
	img_dim = config['img_dim']
	hidden_size1 =	config['hidden_size1']
	hidden_size2 =	config['hidden_size2']
	z_dim =	config['z_dim']
	batch_size = config['batch_size'] 
	num_class =	config['num_class']
	learn_rate = config['learn_rate']
	epoch = config['epoch']
	dropout_keep_prob = config['keep_prob']
	lower_bound_of_stddev = config['lower_bound_of_stddev']
	clip_threshold = config['clip_threshold']

	train_total_data, train_size, _, _, test_data, test_labels = stream.prepare_MNIST_data()


	x_in = tf.placeholder(tf.float32, shape = [None, img_dim], name = 'input_img')
	x_recon = tf.placeholder(tf.float32, shape = [None, img_dim], name = 'trg_img')
	z_in = tf.placeholder(tf.float32, shape = [None, z_dim], name = 'latent_variable')
	keep_prob = tf.placeholder(tf.float32,  name = 'keep_prob')

	## network 
	y, z, loss, neg_marginal_likelihood, KL_divergence = model.autoencoders(x_in, x_recon, img_dim, z_dim, \
		hidden_size1, hidden_size2, keep_prob, lower_bound_of_stddev, clip_threshold)
	train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

	batch_num = int(train_size / batch_size)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : dropout_keep_prob})
		for i in range(epoch):
			if i > 0:
				print('{}-th epoch, negative marginal:{}; kl divergence: {}'.format(i, neg_marg_sum / batch_num, kl_div_sum / batch_num))
			neg_marg_sum, kl_div_sum = 0.0,0.0
			for j in range(batch_num):
				train_data = train_total_data[j*batch_size:(j+1)*batch_size, :img_dim]

				_, neg_marg, kl_div = sess.run([train_op, neg_marginal_likelihood, KL_divergence], feed_dict = {\
					x_in:train_data, \
					x_recon:train_data,\
					keep_prob : dropout_keep_prob\
					 })
				neg_marg_sum += neg_marg
				kl_div_sum += kl_div


if __name__ == '__main__':
	train()












