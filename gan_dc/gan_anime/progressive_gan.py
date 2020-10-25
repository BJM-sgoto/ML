import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt

class ProgressiveGan:
	def __init__(self, img_height=64, img_width=64, z_size=128, img_channel=3, learning_rate=1e-3, num_epoch=50, train_batch_size=40, test_batch_size=40):
		self.img_height = img_height
		self.img_width = img_width
		self.img_channel = img_channel
		self.learning_rate = learning_rate
		self.num_epoch = num_epoch
		self.z_size = z_size
		self.flow_param_values = [0.0,0.0,0.0]
		self.flow_params = []
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		for param in self.flow_param_values:
			self.flow_params.append(tf.Variable(0.0, trainable=False))
	
	def create_dataset(self, image_folder='./image/'):
		images = os.listdir(image_folder)
		inputs = []
		for image in images:
			inputs.append(image_folder + image)
		return {'input': inputs}
	
	def read_flow_params(self, file_path):
		f = open(file_path, 'r')
		s = f.readline()
		f.close()
		params = s.split(' ')
		values = []
		for param in params:
			values.append(float(param))
		return values
		
	def write_flow_params(self, file_path):
		f = open(file_path, 'w')
		values = ''
		for value in self.flow_param_values:
			values += str(value) + ' '
		f.write(values[:-1])
		f.close()
		
	def naive_upsample(self, input_holder, rate):
		input_shape = input_holder.get_shape()
		return tf.image.resize_images(input_holder, 
																	size=(input_shape[1].value*rate, input_shape[2].value*rate),
																	method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	def naive_downsample(self, input_holder, rate):
		input_shape = input_holder.get_shape()
		return tf.image.resize_images(input_holder, 
																	size=(int(input_shape[1].value/rate), int(input_shape[2].value/rate)),
																	method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	
	# flow params : parameters that control the flow from lower layer
	def generate(self, input_holder, training=True):
		with tf.variable_scope('generator'):
			output = tf.reshape(input_holder, shape=[-1, 1, 1, self.z_size])
			
			# generator
			
			# layer 0 output 8 * 8
			output = tf.layers.conv2d_transpose(
								output,
								filters=256,
								kernel_size=(8,8),
								padding='valid',
								use_bias=False)
			output = tf.layers.batch_normalization(output, training=training)
			output = tf.nn.leaky_relu(output)
			# to RGB 
			rgb0 = tf.layers.conv2d(
							output,
							filters=3,
							kernel_size=(1,1),
							use_bias=True)
			rgb0 = self.naive_upsample(rgb0, 8) * (1 - self.flow_params[0])
			
			# layer 1: output 16 * 16
			output = tf.layers.conv2d_transpose(
								output,
								filters=128,
								kernel_size=(3,3),
								strides=(2,2),
								padding='same',
								use_bias=False)
			output = tf.layers.batch_normalization(output, training=training)
			output = tf.nn.leaky_relu(output)
			# to RGB 
			rgb1 = tf.layers.conv2d(
							output,
							filters=3,
							kernel_size=(1,1),
							use_bias=True)
			rgb1 = self.naive_upsample(rgb1, 4) * self.flow_params[0] * (1 - self.flow_params[1])
			
			# layer 2: output 32 * 32
			output = tf.layers.conv2d_transpose(
								output,
								filters=64,
								kernel_size=(3,3),
								strides=(2,2),
								padding='same',
								use_bias=False)
			output = tf.layers.batch_normalization(output, training=training)
			output = tf.nn.leaky_relu(output)
			# to RGB 
			rgb2 = tf.layers.conv2d(
							output,
							filters=3,
							kernel_size=(1,1),
							use_bias=True)
			rgb2 = self.naive_upsample(rgb2, 2) * self.flow_params[0] * self.flow_params[1] * (1 - self.flow_params[2])

			# layer 3: output 64 * 64
			output = tf.layers.conv2d_transpose(
								output,
								filters=32,
								kernel_size=(3,3),
								strides=(2,2),
								padding='same',
								use_bias=False)
			output = tf.layers.batch_normalization(output, training=training)
			output = tf.nn.leaky_relu(output)
			# to RGB 
			output = tf.layers.conv2d(
							output,
							filters=3,
							kernel_size=(1,1),
							use_bias=True)
			rgb3 = output # self.naive_upsample(rgb3, 1)
			rgb3 = rgb3 * self.flow_params[0] * self.flow_params[1] * self.flow_params[2]
			
			# sum all rgb
			output = rgb0 + rgb1 + rgb2 + rgb3
			output = tf.nn.tanh(output)
			return output
	
	def discriminate(self, input_holder, training=True):
		with tf.variable_scope('discriminator'):
			# discriminator
			output = tf.layers.batch_normalization(input_holder)
			
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
			output = output * self.flow_params[2]
			
			feature3 = self.naive_downsample(input_holder, 2)
			feature3 = tf.layers.conv2d(
									feature3,
									filters=64,
									kernel_size=(1,1),
									padding='valid',
									use_bias=False)
			feature3 = feature3 * (1 - self.flow_params[2])
			output = tf.nn.tanh(output	+ feature3)
			
			# layer 1: output size 16 * 16
			output = tf.layers.batch_normalization(output)
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
			output =  output * self.flow_params[1]
			
			feature2 = self.naive_downsample(input_holder, 4)
			feature2 = tf.layers.conv2d(
									feature2,
									filters=128,
									kernel_size=(1,1),
									padding='valid',
									use_bias=False)
			feature2 = feature2 * (1 - self.flow_params[1])
			
			output = tf.nn.tanh(output + feature2)
			
			# layer 2: output size 8 * 8
			output = tf.layers.batch_normalization(output)
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
			output = output * self.flow_params[0]
									
			feature1 = self.naive_downsample(input_holder, 8)
			feature1 = tf.layers.conv2d(
									feature1,
									filters=256,
									kernel_size=(1,1),
									padding='valid',
									use_bias=False)
			feature1 = feature1 * (1 - self.flow_params[0])
			
			output = tf.nn.tanh(feature1 + output)
			output = tf.layers.flatten(output)
			
			output = tf.layers.dense(
								output,
								units=32,
								use_bias=False)
			output = tf.layers.batch_normalization(output)
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
		
		discriminator_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(discriminator_cost, var_list=tf.trainable_variables('discriminator'))
		generator_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(generator_cost, var_list=tf.trainable_variables('generator'))
		
		saver = tf.train.Saver()
		
		session = tf.Session()
		session.run(tf.global_variables_initializer())
		
		if resume:
			self.flow_param_values = self.read_flow_params('./model/flow_params.txt')
			saver.restore(session, './model/model')
		dataset = self.create_dataset(image_folder=image_folder)
		num_data = len(dataset['input'])
		
		for i in range(self.num_epoch):
			random.shuffle(dataset['input'])

			# change flow param
			self.flow_param_values[2] += 0.01
			if self.flow_param_values[2]>1.0:
				self.flow_param_values[2]=1.0
				
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
																		[generator_cost, accuracy, generator_optimizer], 
																		feed_dict={
																			Z: latents, 
																			I: images,
																			self.flow_params[0]:self.flow_param_values[0],
																			self.flow_params[1]:self.flow_param_values[1],
																			self.flow_params[2]:self.flow_param_values[2]})
					print('Generator Loss', generator_loss, 'Accuracy', acc)
				for k in range(1):
					discriminator_loss, acc, _ = session.run(
																			[discriminator_cost, accuracy, discriminator_optimizer], 
																			feed_dict={
																				Z: latents,
																				I: images,
																				self.flow_params[0]:self.flow_param_values[0],
																				self.flow_params[1]:self.flow_param_values[1],
																				self.flow_params[2]:self.flow_param_values[2]})
					print('Discriminator Loss', discriminator_loss, 'Accuracy', acc)
										
			saver.save(session,'./model/model')
			self.write_flow_params('./model/flow_params.txt')
			fig = plt.figure()
			latents = np.random.normal(size=[25, self.z_size])
			out_images = session.run(
										fake_images,
										feed_dict={
											Z: latents,
											self.flow_params[0]:self.flow_param_values[0],
											self.flow_params[1]:self.flow_param_values[1],
											self.flow_params[2]:self.flow_param_values[2]})
			out_images = out_images*127.5+127.5
			for j in range(1,26):
				img = np.uint8(out_images[j-1])
				fig.add_subplot(5,5,j)
				plt.imshow(img)
			fig.savefig('./output_train/{:06d}.jpg'.format(i+128))
		session.close()
		
	def test(self, output_folder):
		Z = tf.placeholder(tf.float32, shape=[None, self.z_size])
		fake_images = self.generate(Z, training=False)
		saver = tf.train.Saver()
		
		session = tf.Session()
		session.run(tf.global_variables_initializer())
		saver.restore(session, './model/model')
		self.flow_param_values = self.read_flow_params('./model/flow_params.txt')
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
		
model = ProgressiveGan(
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
	resume=False)


'''
model.test(
	output_folder='./output/')
'''