import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import random

RANDOM_SEED = 1234
tf.disable_v2_behavior()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

IMG_WIDTH = 96
IMG_HEIGHT = 128
Z_DIM = 100

class Model:
	def __init__(self):
		self.reuse_generator = False
		self.reuse_discriminator = False
		self.kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
		
	def make_dataset(self, image_folder):
		dataset = []
		for image in os.listdir(image_folder):
			dataset.append(image_folder + image)
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def generate(self, z, training=False):
		with tf.variable_scope('generator', reuse=self.reuse_generator):
			x = z
			x = tf.layers.dense(x, units=512 * 16 * 12)
			x = tf.reshape(x, [-1, 16, 12, 512])
			x_out1 = (tf.layers.dense(x, units=3, activation=tf.nn.tanh)+1)*127.5
			x = tf.nn.leaky_relu(x, alpha=0.2)
			# upsample to 16X16
			x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same')
			x_out2 = (tf.layers.dense(x, units=3, activation=tf.nn.tanh)+1)*127.5
			x = tf.nn.leaky_relu(x, 0.2)
			# upsample to 32X32
			x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same')
			x_out3 = (tf.layers.dense(x, units=3, activation=tf.nn.tanh)+1)*127.5
			x = tf.nn.leaky_relu(x, 0.2)
			# upsample to 64X64
			x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, 0.2)
			# generate
			x = tf.layers.conv2d(x, filters=3, kernel_size=7, activation=tf.nn.tanh, padding='same')
			x_out4 = (x + 1) * 127.5
			self.reuse_generator = True
			return x_out1, x_out2, x_out3, x_out4
	
	def discriminate(self, xs, training=False):
		with tf.variable_scope('discriminator', reuse=self.reuse_discriminator):
			x_out1, x_out2, x_out3, x_out4 = xs
			x_out1 = x_out1 / 127.5 - 1
			x_out2 = x_out2 / 127.5 - 1
			x_out3 = x_out3 / 127.5 - 1
			x_out4 = x_out4 / 127.5 - 1
			x = x_out4
			# downsample
			x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, alpha=0.2)
			x = tf.concat([x, x_out3], axis=3)
			# downsample
			x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, alpha=0.2)
			x = tf.concat([x, x_out2], axis=3)
			# downsample
			x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, alpha=0.2)
			tf.concat([x, x_out1], axis=3)
			# discriminate
			x = tf.layers.flatten(x)
			x = tf.layers.dropout(x, 0.4, training=training)
			x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
			x = tf.squeeze(x, 1)
			#x = tf.Print(x,[x], 'Predicted', summarize=100)
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
	
	def train_batch(self, batch, session, real_inputs, fake_input, generator_cost, discriminator_cost, generator_train_op, discriminator_train_op):
		rx1 = []
		rx2 = []
		rx3 = []
		rx4 = []
		RX1, RX2, RX3, RX4 = real_inputs
		Z = fake_input
		for image_file in batch:
			image_4 = cv2.imread(image_file)
			image_3 = cv2.resize(image_4, (int(IMG_WIDTH/2), int(IMG_HEIGHT/2)))
			image_2 = cv2.resize(image_3, (int(IMG_WIDTH/4), int(IMG_HEIGHT/4)))
			image_1 = cv2.resize(image_2, (int(IMG_WIDTH/8), int(IMG_HEIGHT/8)))
			rx1.append(image_1)
			rx2.append(image_2)
			rx3.append(image_3)
			rx4.append(image_4)
		rx1 = np.float32(rx1)
		rx2 = np.float32(rx2)
		rx3 = np.float32(rx3)
		rx4 = np.float32(rx4)
		# train generator
		n_data = len(batch)
		z = np.random.normal(size=[n_data, Z_DIM])	
		gen_loss_val, _ = session.run([generator_cost, generator_train_op], feed_dict={Z: z, RX1: rx1, RX2: rx2, RX3: rx3, RX4: rx4})
		z = np.random.normal(size=[n_data, Z_DIM])	
		gen_loss_val, _ = session.run([generator_cost, generator_train_op], feed_dict={Z: z, RX1: rx1, RX2: rx2, RX3: rx3, RX4: rx4})
		# train discriminator
		dis_loss_val, _ = session.run([discriminator_cost, discriminator_train_op], feed_dict={Z: z, RX1: rx1, RX2: rx2, RX3: rx3, RX4: rx4})
		return gen_loss_val, dis_loss_val
	
	def train(self, num_epochs=100, batch_size=128, image_folder='./image/', output_folder='./output/', model_path='./fashion_mnist/model', resume=False):
		Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
		RX1 = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT/8, IMG_WIDTH/8, 3])
		RX2 = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT/4, IMG_WIDTH/4, 3])
		RX3 = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT/2, IMG_WIDTH/2, 3])
		RX4 = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		
		FX1, FX2, FX3, FX4 = self.generate(Z, training=True)
		X1 = tf.concat([RX1, FX1], axis=0)
		X2 = tf.concat([RX2, FX2], axis=0)
		X3 = tf.concat([RX3, FX3], axis=0)
		X4 = tf.concat([RX4, FX4], axis=0)
		PY = self.discriminate([X1, X2, X3, X4], training=True)
		
		generator_cost, discriminator_cost = self.compute_cost(PY)
		generator_train_op = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(generator_cost, var_list=tf.trainable_variables('generator'))
		discriminator_train_op = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(discriminator_cost, var_list=tf.trainable_variables('discriminator'))
		
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
			
		dataset = self.make_dataset(image_folder=image_folder)
		num_data = len(dataset)
		count_to_draw = 0
		count_to_save = 68
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j + batch_size)
				batch = dataset[j:end_j]
				gen_loss_val, dis_loss_val = self.train_batch(batch, session, [RX1, RX2, RX3, RX4], Z, generator_cost, discriminator_cost, generator_train_op, discriminator_train_op)
				
				count_to_draw += 1
				print('Count {:03d}, Epoch {:02d}, Progress {:05d}, GL {:06f}, DL {:06f}'.format(count_to_draw, i, j, gen_loss_val, dis_loss_val))
				if count_to_draw>=100:
					count_to_draw = 0
					z = np.random.normal(size=[25, Z_DIM])	
					fx = session.run(FX4, feed_dict={Z: z})
					image = np.ones([IMG_HEIGHT * 5 + 10 * 4, IMG_WIDTH * 5 + 10 * 4, 3]) * 255
					for m in range(5):
						for n in range(5):
							image[m*(IMG_HEIGHT + 10): m*(IMG_HEIGHT + 10) + IMG_HEIGHT, n*(IMG_WIDTH + 10): n*(IMG_WIDTH + 10) + IMG_WIDTH] = fx[m*5+n]
					cv2.imwrite(output_folder + 'output_{:06d}.jpg'.format(count_to_save), image)
					count_to_save += 1
					saver.save(session, model_path)
		session.close()

model = Model()

model.train(
	num_epochs=2,
	batch_size=30,
	image_folder='./small_celeba/',
	output_folder='./output_msggan/',
	model_path='./model/model_msggan', 
	resume=True)
