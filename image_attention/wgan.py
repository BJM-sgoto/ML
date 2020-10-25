# reference : https://github.com/taki0112/Self-Attention-GAN-Tensorflow/
# reference : https://en.m.wikipedia.org/wiki/Lipschitz_continuity
import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np
import os
import random
import datetime
#import matplotlib.pyplot as plt

RANDOM_SEED = 1234

IMG_HEIGHT = 128
IMG_WIDTH = 128
Z_DIM = 512

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class Model:
	def __init__(self):
		self.reuse_generator = False
		self.reuse_discriminator = False
		
	def make_dataset(self, image_folder='./image/'):
		dataset = []
		for file_name in os.listdir(image_folder):
			dataset.append(image_folder + file_name)
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
		
	def conv(self, x, layer_depth, kernel, use_spectral_norm=False, use_bias=True, scope='conv'):
		with tf.variable_scope(scope):
			depth = x.shape[3].value
			if use_spectral_norm:
				w = tf.get_variable('kernel', shape=[kernel, kernel, depth, layer_depth], initializer=tf.random_normal_initializer())
				w = self.spectral_norm(w)
				x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
			else:
				x = tf.layers.conv2d(
					x,
					filters=layer_depth,
					kernel_size=kernel,
					strides=1,
					padding='same',
					use_bias=False)
			if use_bias:
				bias = tf.get_variable('bias', [layer_depth], tf.float32, initializer=tf.constant_initializer(0.0))
				x = x + bias
			return x
	
	def deconv(self, x, layer_depth, kernel, use_spectral_norm, use_bias=True, scope='deconv'):
		with tf.variable_scope(scope):
			batch_size = tf.shape(x)[0]
			height = x.shape[1].value
			width = x.shape[2].value
			depth = x.shape[3].value
			if use_spectral_norm:
				w = tf.get_variable('kernel', shape=[kernel, kernel, layer_depth, depth], initializer=tf.random_normal_initializer())
				w = self.spectral_norm(w)
				x = tf.nn.conv2d_transpose(
					x, 
					w, 
					output_shape=[batch_size, height*2, width*2, layer_depth],
					strides=[1,2,2,1], 
					padding='SAME')
			else:
				x = tf.layers.conv2d_transpose(
					x,
					filters=layer_depth,
					kernel_size=kernel,
					strides=2,
					padding='same',
					use_bias=False)
			if use_bias:
				bias = tf.get_variable('bias', [layer_depth], tf.float32, initializer=tf.constant_initializer(0.0))
				x = x + bias
			return x
			
	def dense(self, x, units,use_spectral_norm=False, use_bias=True, scope='dense'):
		with tf.variable_scope(scope):
			depth = x.shape[1].value
			if use_spectral_norm:
				w = tf.get_variable('kernel', shape=[depth, units], initializer=tf.random_normal_initializer())
				w = self.spectral_norm(w)
				x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
			else:
				x =  tf.layers.dense(
					x,
					units=units,
					use_bias=False)
			if use_bias:
				bias = tf.get_variable('bias', [units], tf.float32, initializer=tf.constant_initializer(0.0))
				x = x + bias
			return x
		
	def spectral_norm(self, w, iteration=5):
		w_shape = w.shape
		w_depth = w_shape[-1].value
		w = tf.reshape(w, [-1, w_depth])
		u = tf.get_variable('u', [1, w_depth], initializer=tf.random_normal_initializer(), trainable=False)
		u_hat = u
		v_hat = None
		for i in range(iteration):
			v_ = tf.matmul(u_hat, tf.transpose(w))
			v_hat = tf.nn.l2_normalize(v_)
			
			u_ = tf.matmul(v_hat, w)
			u_hat = tf.nn.l2_normalize(u_)
		# do not train these
		u_hat = tf.stop_gradient(u_hat)
		v_hat = tf.stop_gradient(v_hat)
		
		sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
		with tf.control_dependencies([u.assign(u_hat)]):
			w_norm = w / sigma
			w_norm = tf.reshape(w_norm, w_shape)
		return w_norm
		
	def up_resblock(self, x, layer_depth, use_batch_norm=False, use_spectral_norm=True, training=False, scope='up_resblock'):
		with tf.variable_scope(scope):
			x_init = x
			height = x.shape[1].value
			width = x.shape[2].value
			depth = x.shape[3].value
			
			#x = tf.image.resize(x_init, (height*2, width*2))
			x = self.deconv(x_init, layer_depth=layer_depth, kernel=5, use_spectral_norm=use_spectral_norm, use_bias=True, scope='deconv_res0')
			# res 1
			x = self.conv(x, layer_depth=layer_depth, kernel=3, use_spectral_norm=use_spectral_norm, scope='conv_res1')
			x = tf.nn.leaky_relu(x, 0.2)
			# res 2
			x = self.conv(x, layer_depth=layer_depth, kernel=3, use_spectral_norm=use_spectral_norm, scope='conv_res2')
			x = tf.nn.leaky_relu(x, 0.2)				
			# short cut
			#x_hat = tf.image.resize(x_init, (height*2, width*2))
			x_hat = self.deconv(x_init, layer_depth=layer_depth, kernel=5, use_spectral_norm=use_spectral_norm, use_bias=True, scope='deconv_shortcut')
			x_hat = self.conv(x_hat, layer_depth=layer_depth, kernel=1, use_spectral_norm=use_spectral_norm, scope='conv_shortcut')
			# combine
			x = x + x_hat
			if use_batch_norm:
				x = tf.layers.batch_normalization(x, momentum=0.5, training=training)
			return x
	
	def down_resblock(self, x, layer_depth, use_batch_norm=False, use_spectral_norm=True, training=False, scope='down_resblock'):
		with tf.variable_scope(scope):
			x_init = x
			height = x.shape[1].value
			width = x.shape[2].value
			depth = x.shape[3].value
			
			x = tf.image.resize(x_init, (int(height/2), int(width/2)))
			# res 1
			x = self.conv(x, layer_depth=layer_depth, kernel=3, use_spectral_norm=use_spectral_norm, scope='conv_res1')
			x = tf.nn.leaky_relu(x, 0.2)
			# res 2
			x = self.conv(x, layer_depth=layer_depth, kernel=3, use_spectral_norm=use_spectral_norm, scope='conv_res2')
			x = tf.nn.leaky_relu(x, 0.2)				
			# short cut
			x_hat = tf.image.resize(x_init, (int(height/2), int(width/2)))
			x_hat = self.conv(x_hat, layer_depth=layer_depth, kernel=1, use_spectral_norm=use_spectral_norm, scope='conv_shortcut')
			# combine
			x = x + x_hat
			if use_batch_norm:
				x = tf.layers.batch_normalization(x, training=training)
			return x
	
	def generate(self, z, training=False):
		with tf.variable_scope('generator', reuse=self.reuse_generator):
			x = tf.layers.dense(z, units=Z_DIM*4*4)
			x = tf.reshape(x, [-1, 4, 4, Z_DIM])
			
			layer_depths = [256,128,128,64,64]
			for i, layer_depth in enumerate(layer_depths):
				if i!=len(layer_depths)-1:
					x = self.up_resblock(x, layer_depth, use_batch_norm=True, use_spectral_norm=False, training=training, scope='up_resblock_' + str(i))
				else:
					x = self.up_resblock(x, layer_depth, use_batch_norm=True, use_spectral_norm=False, training=training, scope='up_resblock_' + str(i))
			x = self.conv(x, layer_depth=3, kernel=3, use_spectral_norm=False, scope='conv_synthesis')		
			x = tf.nn.tanh(x)
			return x
			
	def discriminate(self, x, training=False):
		with tf.variable_scope('discriminator', reuse=self.reuse_discriminator):
			layer_depths = [32,32,64,64,32]
			for i, layer_depth in enumerate(layer_depths):
				if i!=len(layer_depths)-1:
					x = self.down_resblock(x, layer_depth, use_batch_norm=True, use_spectral_norm=False, training=training, scope='down_resblock_' + str(i))
				else:
					x = self.down_resblock(x, layer_depth, use_batch_norm=True, use_spectral_norm=False, training=training, scope='down_resblock_' + str(i))
					
			x = tf.layers.flatten(x)
			x = self.dense(
				x, 
				units=1,
				use_bias=False)
			x = tf.squeeze(x, axis=1)
			self.reuse_discriminator = True
			return x
	
	def compute_cost(self, predicted_outputs):
		batch_size = tf.cast(tf.shape(predicted_outputs)[0]/2, dtype=tf.int32)
		discriminator_cost = tf.reduce_mean(predicted_outputs[:batch_size] - predicted_outputs[batch_size:])
		generator_cost = -tf.reduce_mean(predicted_outputs[batch_size:])
		return discriminator_cost,generator_cost
	
	def make_clip_ops(self, min_val=-0.01, max_val=0.01):
		clip_ops = []
		for var in tf.trainable_variables():
			clipped_var = tf.clip_by_value(var, min_val, max_val)
			clip_ops.append(tf.assign(var, clipped_var))
		return clip_ops
	
	def train_on_batch(self, session, batch, tf_random_inputs, tf_images, tf_generator_cost, tf_discriminator_cost, tf_train_generator_op, tf_train_discriminator_op, tf_predicted_outputs, tf_clip_ops):
		images = []
		for file_path in batch:	
			image = Image.open(file_path)
			image = image.convert('RGB')
			image = np.float32(image)
			cut_y, cut_x = np.random.randint(size=2, low=0, high=21)
			image = image[cut_y: cut_y+IMG_HEIGHT, cut_x: cut_x+IMG_WIDTH]
			images.append(image)
		images = np.float32(images)/127.5-1
		random_inputs = np.float32(np.random.normal(size=[len(images), Z_DIM]))
		# train generator
		generator_loss_val, _ = session.run([tf_generator_cost, tf_train_generator_op],
			feed_dict={tf_random_inputs: random_inputs,
				tf_images: images})
		generator_loss_val, _ = session.run([tf_generator_cost, tf_train_generator_op],
			feed_dict={tf_random_inputs: random_inputs,
				tf_images: images})
		
		
		# train discriminator
		discriminator_loss_val, _ = session.run([tf_discriminator_cost, tf_train_discriminator_op],
			feed_dict={tf_random_inputs: random_inputs,
				tf_images: images})
		
		# clip
		session.run(tf_clip_ops)
		return generator_loss_val, discriminator_loss_val
	
	def generate_image(self, session, tf_random_inputs, tf_output_images):
		random_inputs = np.float32(np.random.normal(size=[20, Z_DIM]))
		output_images = session.run(tf_output_images,
			feed_dict={
				tf_random_inputs: random_inputs})
		output_images = (output_images + 1) * 127.5
		print('------------------\n', output_images.shape)
		image = np.ones([4*128 + 3*10, 5*128 + 4*10 ,3], dtype=np.float32) * 255
		for i in range(4):
			for j in range(5):
				image[i*(128+10): i*(128+10) + 128,j*(128+10): j*(128+10)+128] = output_images[i*5+j]
		image = Image.fromarray(np.uint8(image))
		save_time = datetime.datetime.now()
		image.save('test{:02d}_{:02d}_{:02d}.jpg'.format(save_time.hour, save_time.minute, save_time.second))
	
	def train(self, num_epochs=10, batch_size=20, image_folder='./image/', model_path='./model/model', resume=False):
		Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
		RX = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		FX = self.generate(Z, training=True)
		X = tf.concat([RX, FX], axis=0)
		Y = tf.placeholder(tf.float32, shape=[None, 2])
		PY = self.discriminate(X, training=True)
		
		discriminator_cost,generator_cost = self.compute_cost(PY)
		train_generator_op = tf.train.RMSPropOptimizer(5e-5).minimize(generator_cost, var_list=tf.trainable_variables('generator'))
		update_generator_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
		train_generator_op = tf.group([train_generator_op, update_generator_op])
		train_discriminator_op = tf.train.RMSPropOptimizer(5e-5).minimize(discriminator_cost, var_list=tf.trainable_variables('discriminator'))
		update_discriminator_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
		train_discriminator_op = tf.group([train_discriminator_op, update_discriminator_op])
		
		clip_ops = self.make_clip_ops()
		
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		dataset = self.make_dataset(image_folder)
		num_data = len(dataset)
		count_to_save = 0
		
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j + batch_size)
				batch = dataset[j: end_j]
				generator_loss_val, discriminator_loss_val = self.train_on_batch(session, batch, Z, RX, generator_cost, discriminator_cost, train_generator_op, train_discriminator_op, PY, clip_ops)
				print('Epoch {:02d},Progress {:04d},Generator Loss {:06f},Discriminator Loss {:06f}'.format(i, j, generator_loss_val, discriminator_loss_val))
				count_to_save+=1
				if count_to_save>=100:
					self.generate_image(session, Z, FX)
					saver.save(session, model_path)
					count_to_save = 0
		saver.save(session, model_path)
		session.close()
		
model = Model()
model.train(
	num_epochs=10,
	batch_size=10,
	image_folder='./small_cat/',
	model_path='./model/model',
	resume=False)