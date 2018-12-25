def get_config():
	config = {}
	config['img_dim'] = 784
	config['hidden_size1'] = 500
	config['hidden_size2'] = 200
	config['z_dim'] = 10
	config['batch_size'] = 256
	config['num_class'] = 10 
	config['learn_rate'] = 1e-3
	config['epoch'] = 50
	config['keep_prob'] = 0.9 
	config['lower_bound_of_stddev'] = 1e-5
	config['clip_threshold'] = 1e-4

	return config


