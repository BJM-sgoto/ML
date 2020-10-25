import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
tf.reset_default_graph()

IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 7

class Model:
	def __init__(self):
		self.kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()
	
	def instance_normalization(self, X, name='instance_normalization'):
		with tf.variable_scope(name):
			mean = tf.reduce_mean(X, axis=[1,2], keepdims=True)
			X = X - mean
			stddev = tf.sqrt(tf.reduce_mean(tf.square(X), axis=[1,2], keepdims=True)+1e-6)
			depth = X.get_shape()[-1].value
			new_mean = tf.get_variable('new_mean', shape=[depth], dtype=tf.float32)
			new_stddev = tf.get_variable('new_stddev', shape=[depth], dtype=tf.float32)
			X = X / stddev * new_stddev + new_mean
			return X			
	
	def atrous_conv2d(self, X, filters, kernel_size=3, rate=2, use_bias=False, name='atrous_conv2d'):
		with tf.variable_scope(name):
			_, h, w, c = X.get_shape().as_list()
			kernel = tf.get_variable('kernel', shape=[kernel_size, kernel_size, c, filters], dtype=tf.float32,initializer=self.kernel_initializer)
			print('kernel', kernel)
			X = tf.nn.atrous_conv2d(X, filters=kernel, rate=rate, padding='SAME')
			if use_bias:
				bias = tf.get_variable('bias', shape=[kernel_size], dtype=tf.float32, initializer=self.bias_initializer)
				X = tf.add_bias(X, bias)
			return X
	
	def atrous_pooling2d(self, X, rates=[1, 2, 4, 8], name='atrous_pooling2d'):
		Xs = []
		with tf.variable_scope(name):
			for i, rate in enumerate(rates):
				if rate==1:
					Xt = self.atrous_conv2d(X, filters=32, kernel_size=1, rate=1, name='atrous_conv2d_' + str(i))
				else:
					Xt = self.atrous_conv2d(X, filters=32, kernel_size=3, rate=rate, name='atrous_conv2d_' + str(i))
				Xs.append(Xt)
			X = tf.concat(Xs, axis=3)
			return X
			
	def spade(self, X, C, training=False, name='spade'):
		with tf.variable_scope(name):	
			_, h, w, c = X.get_shape().as_list()
			C = tf.image.resize(C, [w, h], 'nearest')
			C = tf.layers.conv2d(C, kernel_size=3, filters=128, padding='same', activation=tf.nn.relu)
			C_mean = tf.layers.conv2d(C, kernel_size=3, filters=c, padding='same')
			C_stddev = tf.layers.conv2d(C, kernel_size=3, filters=c, padding='same', activation=tf.nn.softplus)
			
			X = tf.layers.batch_normalization(X, training=training)
			X = X * C_stddev + C_mean
			return X
	
	def spade_block(self, X, C, filters, training=False, name='spade_block'):
		with tf.variable_scope(name):
			X = self.spade(X, C, training=training)
			X = tf.nn.relu(X)
			X = tf.layers.conv2d(X, filters=filters, kernel_size=3, padding='same')
			'''
			X = self.spade(X, C, training=training)
			X = tf.nn.relu(X)
			X = tf.layers.conv2d(X, filters=filters, kernel_size=3, padding='same')
			'''
			return X
	
	def generate(self, Z, C, training=False, name='generator'):
		with tf.variable_scope(name):
			X = tf.layers.dense(Z, units=256*8*8)
			X = tf.reshape(X, [-1, 8, 8, 256])
			
			layers = [256,256,128,128,64,64]
			for i, layer in enumerate(layers):
				X = self.spade_block(X, C, layer, name='spade_block_' + str(i))
				_, h, w, _ = X.get_shape().as_list()
				if i!=len(layers)-1:
					X = tf.image.resize(X, [h*2, w*2], 'nearest')
			X = tf.layers.conv2d(X, filters=3, kernel_size=3, padding='same', use_bias=False, activation=tf.nn.tanh)
			return X
		
	def discriminate(self, X, C, name='discriminator'):
		with tf.variable_scope(name):
			X = tf.concat([X,C], axis=3)
			

	def segment(self, X, training=False, name='segment'):
		X = X/127.5-1
		with tf.variable_scope(name):
			extractor_layers = [32,64]
			# field : 18
			for layer in extractor_layers:
				X = tf.layers.conv2d(X, filters=layer, kernel_size=3, padding='same', activation=tf.nn.relu)
				X = tf.layers.conv2d(X, filters=layer, kernel_size=3, padding='same', activation=tf.nn.relu)
				X = tf.layers.max_pooling2d(X, strides=2, pool_size=2)
				X = tf.layers.batch_normalization(X, training=training)
			# field : 18 + 4*4 = 34
			X = self.atrous_conv2d(X, filters=128, kernel_size=3, rate=2, use_bias=False, name='atrous_conv2d_1')
			# field : 34 + 8*4 = 66
			X = self.atrous_conv2d(X, filters=128, kernel_size=3, rate=4, use_bias=False, name='atrous_conv2d_2')
			# field : max : 66 + 16*4 = 130
			X = self.atrous_pooling2d(X, rates=[1,2,4,8], name='atrous_pooling2d')
			
			X = tf.layers.dense(X, units=NUM_CLASSES, use_bias=False, activation=tf.nn.sigmoid)
			X = tf.nn.softmax(X, axis=3)
			return X
		
	def compute_cost(self, predicted_segmentation, target_segmentation):
		dictionary = np.eye(NUM_CLASSES)
		target_segmentation = tf.expand_dims(target_segmentation, axis=3)
		target_segmentation = tf.gather_nd(dictionary, target_segmentation)
		cost = tf.reduce_mean(tf.square(target_segmentation - predicted_segmentation))
		return cost
			
	def train(self, image_folder='./image/', segment_folder='./segment/', output_folder='', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3])
		C = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES])
		generator_name='generator'
		PY = self.generate(X, C, training=True, name=generator_name)
		print(PY)
		
model = Model()
model.train()
