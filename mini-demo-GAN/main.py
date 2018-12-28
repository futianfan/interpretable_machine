import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import GAN 

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 200, "batch size")
flags.DEFINE_integer("batch_num", 300, "batch number")
flags.DEFINE_integer("img_dim", 784, "image dimension")
flags.DEFINE_integer("z_dim", 100, "z dimension")
flags.DEFINE_integer("g_hidden_size", 300, "generator hidden size")
flags.DEFINE_integer("d_hidden_size", 50, "discriminator hidden size")

flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam [0.0002]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "../MNIST_data", "Root directory of dataset [data]")

'''
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
'''
FLAGS = flags.FLAGS

#assert FLAGS.epoch == 25
#assert FLAGS.learning_rate == 0.0002

def train(FLAGS):
	assert os.path.exists(FLAGS.data_dir)
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)


	###  data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	### Session
	sess = tf.Session()

	###  model
	model = GAN(sess, FLAGS)
	
	for i in range(FLAGS.epoch): 
		D_loss, G_loss = 0, 0 
		for j in range(FLAGS.batch_num):
		#for j in range(3):
			X_mb, _ = mnist.train.next_batch(FLAGS.batch_size)   ### 200, 784 
			dis_loss, gen_loss = model.train(X_mb)
			D_loss += dis_loss
			G_loss += gen_loss
		print('Epoch {}, Discriminator loss {}, generator loss {}'.format(i, D_loss, G_loss))
	
if __name__ == '__main__':
	train(FLAGS)





