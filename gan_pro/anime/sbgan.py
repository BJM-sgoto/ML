import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import random
Z_DIM = 512
W_DIM = 512
IMG_WIDTH, IMG_HEIGHT = 64, 64

tf.disable_v2_behavior()
tf.reset_default_graph()

class Model:
	def __init__(self):
		self.reuse_generator = False
		self.reuse_discriminator = False
		self.kernel_initializer = tf.random_normal_initializer(mean=0.00, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()		
	
	def make_dataset(self, image_folder='./image/'):
		dataset = []
		for image in os.listdir(image_folder):
			dataset.append(image_folder + image)
		return dataset
	
	def get_constant_z_latent(self):
		seed = 1234
		np.random.seed(seed)
		latent = np.random.normal(size=[1,4,4,512])
		seed = None
		np.random.seed(seed)
		return latent
		
	def adain(self, x, a):
		depth = x.get_shape()[-1].value
		mean_x = tf.reduce_mean(x, axis=[1,2], keepdims=True)
		stddev_x = tf.sqrt(tf.reduce_mean(tf.square(x - mean_x), axis=[1,2], keepdims=True) + 1e-6)
		a = tf.reshape(a, [-1, 1, 1, a.get_shape()[-1].value])
		mean_style = tf.layers.dense(a, units=depth, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)
		stddev_style = tf.layers.dense(a, units=depth, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer, activation=tf.nn.softplus)
		x = (x - mean_x) / stddev_x * mean_style + stddev_style
		return x
	
	def generate(self, z, w):
		x = z
		with tf.variable_scope('generator', reuse=self.reuse_generator):
			for i in range(4):
				w = tf.layers.dense(w, units=W_DIM, kernel_initializer=self.kernel_initializer)
			
			layer_depths = [512, 256,128,64,32]
			for i, layer_depth in enumerate(layer_depths):
				if i!=0:
					_, x_height, x_width, _ = x.get_shape().as_list()
					x = tf.image.resize(x, (x_width*2, x_height*2), 'bilinear')
					x = tf.layers.conv2d(x, filters=layer_depth, kernel_size=3, padding='same')
				
				# add noise
				noise = tf.random.normal(tf.shape(x), mean=0.00, stddev=0.02,dtype=tf.float32)
				x = x + noise
				
				# adain
				print('--------------\n', x, w)
				x = self.adain(x, w)
				
				# conv2d
				x = tf.layers.conv2d(x, filters=layer_depth, kernel_size=3, padding='same')
				
				# add noise
				noise = tf.random.normal(tf.shape(x), mean=0.00, stddev=0.02,dtype=tf.float32)
				x = x + noise
				
				# adain
				print('--------------\n', x, w)
				x = self.adain(x, w)
			
			x = tf.layers.dense(x, units=3, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer, activation=tf.nn.tanh)
			x = (x + 1) * 127.5
			self.reuse_generator = True
			return x
	
	def discriminate(self, x):
		with tf.variable_scope('discriminator', reuse=self.reuse_discriminator):
			x = x / 127.5 - 1
			layer_depths = [64,128,256,512]
			for i, layer_depth in enumerate(layer_depths):
				x = tf.layers.conv2d(x, filters=layer_depth, kernel_size=3, padding='same')
				x = tf.layers.conv2d(x, filters=layer_depth, kernel_size=3, strides=2, padding='same')
			x = tf.layers.flatten(x)
			x = tf.layers.dense(x, units=1, use_bias=False, kernel_initializer=self.kernel_initializer, activation=tf.nn.sigmoid)
			self.reuse_discriminator = True
			return x
		
	def compute_cost(self, predicted_outputs):
		batch_size = tf.shape(predicted_outputs)[0]
		half_batch = tf.cast(batch_size/2, dtype=tf.int32)
		
		fake_target_outputs = tf.random.uniform(minval=0.8, maxval=1.0, dtype=tf.float32, shape=[half_batch])
		
		generator_cost = -tf.reduce_mean(fake_target_outputs * tf.log(predicted_outputs[half_batch:] + 1e-6) + (1 - fake_target_outputs) * tf.log(1 - predicted_outputs[half_batch:] + 1e-6))
		
		true_target_outputs = tf.concat(
		[tf.random.uniform(minval=0.8, maxval=1.0, dtype=tf.float32, shape=[half_batch]),
		tf.random.uniform(minval=0.0, maxval=0.2, dtype=tf.float32, shape=[half_batch])], axis=0)
		
		discriminator_cost = -tf.reduce_mean(true_target_outputs * tf.log(predicted_outputs + 1e-6) + (1 - true_target_outputs) * tf.log(1 - predicted_outputs + 1e-6))
		return generator_cost, discriminator_cost

	def train(self, num_steps=100, batch_size=128, image_folder='./image/', output_folder='./output/', model_path='./model/model', resume=False):
		Z = tf.placeholder(tf.float32, shape=[None, 4, 4, Z_DIM])
		W = tf.placeholder(tf.float32, shape=[None, Z_DIM])
		FX = self.generate(Z, W)
		RX = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		X = tf.concat([FX, RX], axis=0)
		PY = self.discriminate(X)
		
		generator_cost, discriminator_cost = self.compute_cost(PY)
		generator_train_op = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(generator_cost, var_list=tf.trainable_variables('generator'))
		generator_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
		generator_train_op = tf.group([generator_train_op, generator_update_op])
		discriminator_train_op = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(discriminator_cost, var_list=tf.trainable_variables('discriminator'))
		discriminator_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
		discriminator_train_op = tf.group([discriminator_train_op, discriminator_update_op])
		
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
			
		# fake latent
		z = self.get_constant_z_latent()
		count_to_save = 0
		count_to_draw = 0
		dataset = self.make_dataset(image_folder=image_folder)
		for i in range(num_steps):
			batch = random.sample(dataset, batch_size)
			rx = []
			for image_file in batch:
				image = cv2.imread(image_file)
				image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
				rx.append(image)
			rx = np.float32(rx)
			# train generator
			w = np.random.normal(size=[batch_size, Z_DIM])
			gen_loss_val, _ = session.run([generator_cost, generator_train_op], feed_dict={Z: z, RX: rx, W: w})
			w = np.random.normal(size=[batch_size, Z_DIM])	
			gen_loss_val, _ = session.run([generator_cost, generator_train_op], feed_dict={Z: z, RX: rx, W: w})
			# train discriminator
			dis_loss_val, _ = session.run([discriminator_cost, discriminator_train_op], feed_dict={Z: z, RX: rx, W: w})
			count_to_draw += 1
			print('Step {:03d}, GL {:06f}, DL {:06f}'.format(i, gen_loss_val, dis_loss_val))
			if count_to_draw>=100:
				count_to_draw = 0
				w = np.random.normal(size=[25, Z_DIM])
				fx = session.run(FX, feed_dict={Z: z, W: w})
				image = np.ones([IMG_HEIGHT * 5 + 10 * 4, IMG_WIDTH * 5 + 10 * 4, 3]) * 255
				for m in range(5):
					for n in range(5):
						image[m*(IMG_HEIGHT + 10): m*(IMG_HEIGHT + 10) + IMG_HEIGHT, n*(IMG_WIDTH + 10): n*(IMG_WIDTH + 10) + IMG_WIDTH] = fx[m*5+n]
				cv2.imwrite(output_folder + 'output_{:06d}.jpg'.format(count_to_save), image)
				count_to_save += 1
		
			
model = Model()
model.train(
	num_steps=100,
	batch_size=5,
	image_folder='./image/',
	output_folder='./output/',
	model_path='./model/model',
	resume=False)