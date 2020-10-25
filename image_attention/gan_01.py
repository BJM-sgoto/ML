# reference : https://github.com/taki0112/Self-Attention-GAN-Tensorflow/
# reference : https://en.m.wikipedia.org/wiki/Lipschitz_continuity
#%tensorflow_version 1.x
import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np
import os
import random
#import matplotlib.pyplot as plt

RANDOM_SEED = 1234

IMG_HEIGHT = 64
IMG_WIDTH = 64
Z_DIM = 128

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class Model:
	def __init__(self):
		self.reuse_generator = False
		self.reuse_discriminator = False
		self.kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.bias_initializer = tf.constant_initializer(value=0)
		
		
	def make_dataset(self, image_folder='./image/'):
		dataset = []
		for file_name in os.listdir(image_folder):
			dataset.append(image_folder + file_name)
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
		
	def spectral_norm(self, w, iteration=1):
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
		
	def deconv(self, x, filter, kernel, stride, use_spectral_norm=True, use_batch_norm=True, use_drop=False, training=False, scope='deconv'):
		with tf.variable_scope(scope):
			if use_spectral_norm:
				x_shape = x.shape
				w = tf.get_variable('kernel', shape=[kernel, kernel, x_shape[-1].value, filter], initializer=self.kernel_initializer)
				w = self.spectral_norm(w)
				w = tf.nn.conv2d_transpose(
					x,
					filters=w,
					output_shape=[tf.shape(x)[0], x_shape[1].value * 2, x_shape[2].value * 2, filter],
					strides=stride,
					padding='SAME')
			else:
				x = tf.layers.conv2d_transpose(
					x,
					filters=filter,
					kernel_size=kernel,
					strides=stride,
					padding='same',
					kernel_initializer=self.kernel_initializer,
					use_bias=False)
			if use_batch_norm:
				x = tf.layers.batch_normalization(x, training=training)
			x = tf.nn.relu(x)
			if use_drop:
				x = tf.layers.dropout(x, rate=0.5, training=training)
			print('Output of deconv', x)
			return x
		
	def generate(self, z, training=False):
		with tf.variable_scope('generator', reuse=self.reuse_generator):
			x = tf.layers.dense(z, units=Z_DIM * 4 * 4, kernel_initializer=self.kernel_initializer)
			x = tf.reshape(x, [-1, 4, 4, Z_DIM])
			layers = [
				{'filter': 512, 'kernel': 4, 'stride': 1, 'use_spectral_norm':False, 'use_drop': True, 'use_batch_norm': True},
				{'filter': 256, 'kernel': 4, 'stride': 2, 'use_spectral_norm':False, 'use_drop': True, 'use_batch_norm': True},
				{'filter': 128, 'kernel': 4, 'stride': 2, 'use_spectral_norm':False, 'use_drop': False, 'use_batch_norm': True},
				{'filter': 64, 'kernel': 4, 'stride': 2, 'use_spectral_norm':False, 'use_drop': False, 'use_batch_norm': True},
				{'filter': 32, 'kernel': 4, 'stride': 2, 'use_spectral_norm':False, 'use_drop': False, 'use_batch_norm': True}]
			for i, layer in enumerate(layers):
				x = self.deconv(
					x, 
					filter=layer['filter'],
					kernel=layer['kernel'],
					stride=layer['stride'],
					use_spectral_norm=layer['use_spectral_norm'],
					use_drop=layer['use_drop'],
					use_batch_norm=layer['use_batch_norm'],
					training=training,
					scope='deconv_' + str(i))
			x = tf.layers.dense(x, units=3, kernel_initializer=self.kernel_initializer)
			x = tf.nn.tanh(x)
			return x
	
	def conv(self, x, filter, kernel, stride, use_spectral_norm=True, use_batch_norm=True, training=False, scope='conv'):
		with tf.variable_scope(scope):
			if use_spectral_norm:
				w = tf.get_variable('kernel', shape=[kernel, kernel, x.shape[-1].value, filter], initializer=self.kernel_initializer)
				w = self.spectral_norm(w)
				x = tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding='SAME')
			else:
				x = tf.layers.conv2d(
					x,
					filters=filter,
					kernel_size=kernel,
					strides=stride,
					use_bias=False,
					padding='same',
					kernel_initializer=self.kernel_initializer)
			if use_batch_norm:
				x = tf.layers.batch_normalization(x, training=training)
			x = tf.nn.leaky_relu(x, 0.2)
			print('Output of conv', x)
			return x
			
	def dense(self, x, units, use_spectral_norm=True, use_batch_norm=True, training=False, scope='dense'):
		with tf.variable_scope(scope):
			if use_spectral_norm:
				w = tf.get_variable('kernel', shape=[x.shape[-1].value, units], initializer=self.kernel_initializer)
				w = self.spectral_norm(w)
				x = tf.matmul(x, w)
			else:
				x = tf.layers.dense(
					x,
					units=units,
					kernel_initializer=self.kernel_initializer)
			if use_batch_norm:
				x = tf.layers.batch_normalization(x, training=training)
			return x
			
	def discriminate(self, x, training=False):
		with tf.variable_scope('discriminator', reuse=self.reuse_discriminator):
			layers = [
				{'filter': 32, 'kernel': 4, 'stride': 1, 'use_spectral_norm': True, 'use_batch_norm': True},
				{'filter': 64, 'kernel': 4, 'stride': 2, 'use_spectral_norm': True, 'use_batch_norm': True},
				{'filter': 128, 'kernel': 4, 'stride': 2, 'use_spectral_norm': True, 'use_batch_norm': True},
				{'filter': 256, 'kernel': 4, 'stride': 2, 'use_spectral_norm': True, 'use_batch_norm': True}]
			for i, layer in enumerate(layers):
				x = self.conv(
					x, 
					filter=layer['filter'],
					kernel=layer['kernel'],
					stride=layer['stride'],
					use_spectral_norm=layer['use_spectral_norm'],
					use_batch_norm=layer['use_batch_norm'],
					training=training,
					scope='conv_' + str(i))
			x = tf.layers.flatten(x)
			x = self.dense(x, 1, use_spectral_norm=True, use_batch_norm=False, training=training, scope='dense_0')
			x = tf.squeeze(x, axis=1)
			x = tf.nn.sigmoid(x)
			return x
	
	def compute_cost(self, predicted_outputs):
		batch_size = tf.cast(tf.shape(predicted_outputs)[0]/2, dtype=tf.int32)
		# discriminator cost 1
		target_outputs = tf.random_uniform([batch_size], minval=0.800, maxval=0.999)
		noise = tf.random_uniform([batch_size], minval=0.000, maxval=1.000)
		target_outputs = tf.where(tf.greater(noise, 0.95), tf.random_uniform([batch_size], minval=0.001, maxval=0.200), target_outputs)
		discriminator_cost_1 = - tf.reduce_mean(target_outputs*tf.log(predicted_outputs[: batch_size]+1e-6) + (1-target_outputs)*tf.log(1-predicted_outputs[: batch_size]+1e-6))
		# discriminator cost 2
		noise = tf.random_uniform([batch_size], minval=0.000, maxval=1.000)
		target_outputs = tf.where(tf.less(noise, 0.05), tf.random_uniform([batch_size], minval=0.800, maxval=0.999), target_outputs)
		discriminator_cost_2 = - tf.reduce_mean(target_outputs*tf.log(predicted_outputs[batch_size :]+1e-6) + (1-target_outputs)*tf.log(1 - predicted_outputs[batch_size :] + 1e-6))
		# generator cost
		fake_outputs = tf.random_uniform([batch_size], minval=0.800, maxval=0.999)
		noise = tf.random_uniform([batch_size], minval=0.000, maxval=1.000)
		fake_outputs = tf.where(tf.less(noise, 0.025), tf.random_uniform([batch_size], minval=0.001, maxval=0.200), fake_outputs)
		generator_cost = -tf.reduce_mean(fake_outputs * tf.log(predicted_outputs[batch_size:]+1e-6) + (1 - fake_outputs) * tf.log(1 - predicted_outputs[batch_size:] + 1e-6))
		return discriminator_cost_1, discriminator_cost_2, generator_cost
				
	def train_on_batch(self, session, batch, tf_random_inputs, tf_images, tf_generator_cost, tf_discriminator_cost_1, tf_discriminator_cost_2, tf_train_generator_op, tf_train_discriminator_op_1, tf_train_discriminator_op_2, tf_predicted_outputs):
		images = []
		for file_path in batch:	
			image = Image.open(file_path)
			image = image.convert('RGB')
			image = np.float32(image)
			images.append(image)
		images = np.float32(images)/127.5-1
		random_inputs = np.float32(np.random.normal(size=[len(images), Z_DIM]))
		# train generator twice
		generator_loss_val, _  = session.run([tf_generator_cost, tf_train_generator_op],
			feed_dict={tf_random_inputs: random_inputs,
				tf_images: images})
		generator_loss_val, _ = session.run([tf_generator_cost, tf_train_generator_op],
			feed_dict={tf_random_inputs: random_inputs,
				tf_images: images})
				
		# train discriminator on real data
		discriminator_loss_val_1, _  = session.run([tf_discriminator_cost_1, tf_train_discriminator_op_1],
			feed_dict={tf_random_inputs: random_inputs,
				tf_images: images})
		
		# train discriminator on fake data
		discriminator_loss_val_2, _ = session.run([tf_discriminator_cost_2, tf_train_discriminator_op_2],
			feed_dict={tf_random_inputs: random_inputs,
				tf_images: images})
		return generator_loss_val, discriminator_loss_val_1, discriminator_loss_val_2
	
	def generate_image(self, session, tf_random_inputs, tf_output_images, output_path):
		random_inputs = np.float32(np.random.normal(size=[20, Z_DIM]))
		output_images = session.run(tf_output_images,
			feed_dict={
				tf_random_inputs: random_inputs})
		output_images = (output_images + 1) * 127.5
		print('------------------\n', output_images.shape)
		image = np.ones([4*IMG_HEIGHT + 3*10, 5*IMG_WIDTH + 4*10 ,3], dtype=np.float32) * 255
		for i in range(4):
			for j in range(5):
				image[i*(IMG_HEIGHT+10): i*(IMG_HEIGHT+10) + IMG_HEIGHT,j*(IMG_WIDTH+10): j*(IMG_WIDTH+10)+IMG_WIDTH] = output_images[i*5+j]
		image = Image.fromarray(np.uint8(image))
		image.save(output_path)
	
	def train(self, num_epochs=10, batch_size=20, image_folder='./image/', output_folder='./output/', model_path='./model/model', resume=False):
		Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
		RX = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		FX = self.generate(Z, training=True)
		X = tf.concat([RX, FX], axis=0)
		Y = tf.placeholder(tf.float32, shape=[None, 2])
		PY = self.discriminate(X, training=True)
		
		discriminator_cost_1, discriminator_cost_2, generator_cost = self.compute_cost(PY)
		
		adam_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
		train_generator_op = adam_optimizer.minimize(generator_cost, var_list=tf.trainable_variables('generator'))
		update_generator_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
		train_generator_op = tf.group([train_generator_op, update_generator_op])
		
		update_discriminator_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
		
		train_discriminator_op_1 = adam_optimizer.minimize(discriminator_cost_1, var_list=tf.trainable_variables('discriminator'))
		train_discriminator_op_1 = tf.group([train_discriminator_op_1, update_discriminator_op])
		
		train_discriminator_op_2 = adam_optimizer.minimize(discriminator_cost_2, var_list=tf.trainable_variables('discriminator'))
		train_discriminator_op_2 = tf.group([train_discriminator_op_2, update_discriminator_op])
		
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		dataset = self.make_dataset(image_folder)
		num_data = len(dataset)
		count_to_save = 0
		count_to_draw = 0
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j + batch_size)
				batch = dataset[j: end_j]
				generator_loss_val, discriminator_loss_val_1, discriminator_loss_val_2 = self.train_on_batch(session, batch, Z, RX, generator_cost, discriminator_cost_1, discriminator_cost_2, train_generator_op, train_discriminator_op_1, train_discriminator_op_2, PY)
				print('Epoch {:02d},Progress {:04d},G Loss {:06f},D1 Loss {:06f},D2 Loss {:06f}'.format(i, j, generator_loss_val, discriminator_loss_val_1, discriminator_loss_val_2))
				count_to_save+=1
				if count_to_save>=100:
					self.generate_image(session, Z, FX, output_path=output_folder+'test_{:06d}.jpg'.format(count_to_draw))
					saver.save(session, model_path)
					count_to_save = 0
					count_to_draw+=1
		saver.save(session, model_path)
		session.close()
		
#base_folder = '/content/gdrive/My Drive/machine_learning_data/'
base_folder = './'
model = Model()
model.train(
	num_epochs=20,
	batch_size=10,
	image_folder=base_folder + 'cat64/',
	output_folder=base_folder + 'cat64_train_output/',
	model_path=base_folder + 'cat64_model/model',
	resume=False)