import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt

class Gan:
	def __init__(self, img_height=64, img_width=64, z_size=128, img_channel=3, learning_rate=1e-3, num_epoch=50, train_batch_size=40, test_batch_size=40):
		self.img_height = img_height
		self.img_width = img_width
		self.img_channel = img_channel
		self.learning_rate = learning_rate
		self.num_epoch = num_epoch
		self.z_size = z_size
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
	
	def create_dataset(self, image_folder='./image/'):
		images = os.listdir(image_folder)
		inputs = []
		for image in images:
			inputs.append(image_folder + image)
		return {'input': inputs}
	
	# flow params : parameters that control the flow from lower layer
	def generate(self, input_holder, training=True):
		with tf.variable_scope('generator'):
			output = tf.reshape(input_holder, shape=[-1, 1, 1, self.z_size])
			
			# generator
			
			# layer 0 output 8*8
			output = tf.layers.conv2d_transpose(
								output,
								filters=256,
								kernel_size=(8,8),
								padding='valid')
			output = tf.layers.batch_normalization(
								output,
								momentum=0.5, 
								training=training)
			output = tf.nn.leaky_relu(output)
			
			# layer 1: output 16 * 16
			output = tf.layers.conv2d_transpose(
								output,
								filters=128,
								kernel_size=(4,4),
								strides=(2,2),
								padding='same')
			output = tf.layers.batch_normalization(
								output,
								momentum=0.5, 
								training=training)
			output = tf.nn.leaky_relu(output)
			
			# layer 2: output 32 * 32
			output = tf.layers.conv2d_transpose(
								output,
								filters=64,
								kernel_size=(4,4),
								strides=(2,2),
								padding='same')
			output = tf.layers.batch_normalization(
								output,
								momentum=0.5, 
								training=training)
			output = tf.nn.leaky_relu(output)
			
			# additional conv ??
			output = tf.layers.conv2d(
							output,
							filters=32,
							kernel_size=(3,3),
							padding='same')
			output = tf.layers.batch_normalization(
								output,
								momentum=0.5, 
								training=training)
			output = tf.nn.leaky_relu(output)
			
			# layer 3: output 64 * 64
			output = tf.layers.conv2d_transpose(
								output,
								filters=3,
								kernel_size=(4,4),
								strides=(2,2),
								padding='same')
			
			output = tf.nn.tanh(output)
			return output
	
	def discriminate(self, input_holder, training=True):
		with tf.variable_scope('discriminator'):
			# discriminator
			output = tf.layers.batch_normalization(
								input_holder,
								momentum=0.5, 
								training=training)
			
			# layer 0: output 32 * 32 :  associated with generator layer 2
			output = tf.layers.conv2d(
									output,
									filters=64,
									kernel_size=(3,3),
									padding='same',
									use_bias=False)
			output = tf.layers.max_pooling2d(
									output,
									pool_size=(2,2),
									strides=(2,2))
			output = tf.nn.leaky_relu(output)
			
			# layer 1: output size 16 * 16
			output = tf.layers.batch_normalization(
								output,
								momentum=0.5, 
								training=training)
			output = tf.layers.conv2d(
									output,
									filters=128,
									kernel_size=(3,3),
									padding='same',
									use_bias=False)
			output = tf.layers.max_pooling2d(
									output,
									pool_size=(2,2),
									strides=(2,2))
			output = tf.nn.leaky_relu(output)
			
			# layer 2: output size 8 * 8
			output = tf.layers.batch_normalization(
								output,
								momentum=0.5, 
								training=training)
			output = tf.layers.conv2d(
									output,
									filters=256,
									kernel_size=(3,3),
									padding='same',
									use_bias=False)
			output = tf.layers.max_pooling2d(
									output,
									pool_size=(2,2),
									strides=(2,2))
			output = tf.nn.leaky_relu(output)
			output = tf.layers.flatten(output)
			
			output = tf.layers.dense(
								output,
								units=32,
								use_bias=False)
			output = tf.layers.batch_normalization(
								output,
								momentum=0.5, 
								training=training)
			output = tf.nn.leaky_relu(output)
			output = tf.layers.dense(
								output,
								units=1,
								use_bias=True,
								activation=tf.nn.sigmoid)
			output = tf.squeeze(output, axis=1)
			return output
	
	def compute_cost(self,label):
		batch_size = tf.cast(tf.shape(label)[0]/2, dtype=tf.int32)
		true_label = tf.concat(
									[tf.random_uniform([batch_size], minval=0.800, maxval=0.999),
									tf.random_uniform([batch_size], minval=0.001, maxval=0.200)],
									axis=0)
		discriminator_cost = tf.reduce_mean(tf.keras.backend.binary_crossentropy(true_label, label))
		fake_label = tf.random_uniform([batch_size], minval=0.800, maxval=0.999)
		generator_cost = tf.reduce_mean(tf.keras.backend.binary_crossentropy(fake_label, label[batch_size: 2*batch_size]))
		return discriminator_cost,generator_cost
	
	def compute_accuracy(self, label):
		batch_size = tf.cast(tf.shape(label)[0]/2, dtype=tf.int32)
		true_label = tf.concat(
									[tf.ones([batch_size]),
									tf.zeros([batch_size])],
									axis=0)
		true_label = tf.cast(tf.equal(tf.greater(label, 0.5), tf.greater(true_label, 0.5)), dtype=tf.float32)
		return tf.reduce_mean(true_label)
	
	def train(self, image_folder='./image/',resume=True):
		Z = tf.placeholder(tf.float32, shape=[None, self.z_size])
		I = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channel])
		fake_images = self.generate(Z, training=True)
		images = tf.concat([I, fake_images], axis=0)
		labels = self.discriminate(images, training=True)
		discriminator_cost, generator_cost = self.compute_cost(labels)
		accuracy = self.compute_accuracy(labels)
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
		discriminator_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(discriminator_cost, var_list=tf.trainable_variables('discriminator'))
		discriminator_opt = tf.group([update_ops, discriminator_opt])
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
		generator_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(generator_cost, var_list=tf.trainable_variables('generator'))
		generator_opt = tf.group([update_ops, generator_opt])
		
		saver = tf.train.Saver()
		
		session = tf.Session()
		
		if resume:
			saver.restore(session, './model/model')
		else:
			session.run(tf.global_variables_initializer())
			
		dataset = self.create_dataset(image_folder=image_folder)
		num_data = len(dataset['input'])
		
		for i in range(self.num_epoch):
			random.shuffle(dataset['input'])
				
			for j in range(0, num_data,self.train_batch_size):
				end_j = min(num_data, j+self.train_batch_size)
				images = []
				
				for k in range(j, end_j):
					images.append(np.float32(Image.open(dataset['input'][k])))
				images = np.float32(images)/127.5 - 1
				latents = np.random.normal(size=[end_j - j,self.z_size])
				print('-------------\nEpoch', i, 'Progress', j)
				for k in range(3):
					generator_loss, acc, _ = session.run(
																		[generator_cost, accuracy, generator_opt], 
																		feed_dict={
																			Z: latents, 
																			I: images})
					print('Generator Loss', generator_loss, 'Accuracy', acc)
				for k in range(1):
					discriminator_loss, acc, _ = session.run(
																			[discriminator_cost, accuracy, discriminator_opt], 
																			feed_dict={
																				Z: latents,
																				I: images})
					print('Discriminator Loss', discriminator_loss, 'Accuracy', acc)
										
			saver.save(session,'./model/model')
			fig = plt.figure()
			latents = np.random.normal(size=[25, self.z_size])
			out_images = session.run(
										fake_images,
										feed_dict={
											Z: latents})
			out_images = out_images*127.5+127.5
			for j in range(1,26):
				img = np.uint8(out_images[j-1])
				fig.add_subplot(5,5,j)
				plt.imshow(img)
			fig.savefig('./output_train/{:06d}.jpg'.format(i + 8))
		session.close()
		
	def test(self, output_folder):
		Z = tf.placeholder(tf.float32, shape=[None, self.z_size])
		fake_images = self.generate(Z, training=False)
		saver = tf.train.Saver()
		
		session = tf.Session()
		session.run(tf.global_variables_initializer())
		saver.restore(session, './model/model')
		count = 0
		for i in range(10):
			latents = np.random.normal(size=[self.test_batch_size, self.z_size])
			out_images = session.run(
										fake_images, 
										feed_dict={
											Z: latents,
											self.flow_params[0]:self.flow_param_values[0],
											self.flow_params[1]:self.flow_param_values[1],
											self.flow_params[2]:self.flow_param_values[2]})
			out_images = np.uint8(out_images*127.5+127.5)
			for j in range(len(out_images)):
				img = Image.fromarray(out_images[j])
				img.save(output_folder + '{:06d}.jpg'.format(count))
				count += 1
		session.close()
		
model = Gan(
	img_height=64, 
	img_width=64, 
	z_size=128, 
	img_channel=3, 
	learning_rate=1e-4, 
	num_epoch=100,
	train_batch_size=80,
	test_batch_size=80)

model.train(
	image_folder='./faces/',
	resume=True)