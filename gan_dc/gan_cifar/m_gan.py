# small test on fashion mnist
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

IMG_WIDTH = 32
IMG_HEIGHT = 32
Z_DIM = 100

class Model:
	def __init__(self):
		self.reuse_generator = False
		self.reuse_discriminator = False
		
	def save_dataset(self, dataset_path, image_file='./image.npy', label_file='./label.npy'):
		f = open(dataset_path, 'r')
		s = f.readline().strip()
		s = f.readline().strip()
		labels = []
		images = []
		count = 0
		while s:
			temp = np.uint8(eval(s))
			labels.append(temp[0])
			images.append(temp[1:])
			s = f.readline().strip()
			count+=1
			print(count)
		labels = np.uint8(labels)
		images = np.uint8(images)
		np.save(image_file, images)
		np.save(label_file, labels)
		f.close()
		
	def make_dataset(self, image_folder):
		dataset = []
		for image in os.listdir(image_folder):
			dataset.append(image_folder + image)
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def generate(self, z, training=False):
		with tf.variable_scope('generator', reuse=self.reuse_generator):
			x = tf.layers.dense(z, units=512 * 8 * 8)
			x = tf.nn.leaky_relu(x, alpha=0.2)
			x = tf.reshape(x, [-1, 8, 8, 512])
			# upsample to 16X16
			x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, 0.2)
			# upsample to 32X32
			x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, 0.2)
			# generate
			x = tf.layers.conv2d(x, filters=3, kernel_size=7, activation=tf.nn.tanh, padding='same')
			x = (x + 1) * 127.5
			self.reuse_generator = True
			return x
	
	def discriminate(self, x, training=False):
		with tf.variable_scope('discriminator', reuse=self.reuse_discriminator):
			x = x / 127.5 - 1
			
			# downsample
			x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, alpha=0.2)
			
			# downsample
			x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, alpha=0.2)
			
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
		#generator_cost = tf.keras.losses.binary_crossentropy(fake_target_outputs, predicted_outputs[half_batch:],from_logits=False)
		generator_cost = -tf.reduce_mean(fake_target_outputs * tf.log(predicted_outputs[half_batch:] + 1e-6) + (1 - fake_target_outputs) * tf.log(1 - predicted_outputs[half_batch:] + 1e-6))
		true_target_outputs = tf.concat(
		[tf.random.uniform(minval=0.8, maxval=1.0, dtype=tf.float32, shape=[half_batch]),
		tf.random.uniform(minval=0.0, maxval=0.2, dtype=tf.float32, shape=[half_batch])], axis=0)
		#discriminator_cost = tf.keras.losses.binary_crossentropy(true_target_outputs,predicted_outputs,from_logits=False)
		discriminator_cost = -tf.reduce_mean(true_target_outputs * tf.log(predicted_outputs + 1e-6) + (1 - true_target_outputs) * tf.log(1 - predicted_outputs + 1e-6))
		return generator_cost, discriminator_cost
	
	def train(self, num_epochs=100, batch_size=128, image_folder='./image/', output_folder='./output/', model_path='./fashion_mnist/model', resume=False):
		Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
		RX = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		FX = self.generate(Z, training=True)
		X = tf.concat([RX, FX], axis=0)
		PY = self.discriminate(X, training=True)
		
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
			
		dataset = self.make_dataset(image_folder=image_folder)
		num_data = len(dataset)
		count_to_draw = 0
		count_to_save = 0
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j + batch_size)
				rx = []
				for k in range(j, end_j):
					image = cv2.imread(dataset[k])
					rx.append(image)
				rx = np.float32(rx)
				z = np.random.normal(size=[end_j - j, Z_DIM])	
				# train generator
				gen_loss_val, _ = session.run([generator_cost, generator_train_op], feed_dict={Z: z, RX: rx})
				# train discriminator
				dis_loss_val, _ = session.run([discriminator_cost, discriminator_train_op], feed_dict={Z: z, RX: rx})
				count_to_draw += 1
				print('Count {:03d}, Epoch {:02d}, Progress {:05d}, GL {:06f}, DL {:06f}'.format(count_to_draw, i, j, gen_loss_val, dis_loss_val))
				if count_to_draw>=100:
					count_to_draw = 0
					z = np.random.normal(size=[25, Z_DIM])	
					fx = session.run(FX, feed_dict={Z: z})
					image = np.ones([IMG_HEIGHT * 5 + 10 * 4, IMG_WIDTH * 5 + 10 * 4, 3]) * 255
					for m in range(5):
						for n in range(5):
							image[m*(IMG_HEIGHT + 10): m*(IMG_HEIGHT + 10) + IMG_HEIGHT, n*(IMG_WIDTH + 10): n*(IMG_WIDTH + 10) + IMG_WIDTH] = fx[m*5+n]
					cv2.imwrite(output_folder + 'test1_{:06d}.jpg'.format(count_to_save), image)
					count_to_save += 1
			saver.save(session, model_path)
		session.close()
				

model = Model()
'''
model.save_dataset(
	dataset_path='./fashionmnist/fashion-mnist_test.csv', 
	image_file='./fashionmnist/test_image.npy', 
	label_file='./fashionmnist/test_label.npy')
'''
model.train(
	num_epochs=200,
	batch_size=50,
	image_folder='./train/9/',
	output_folder='./train_output/9/',
	model_path='./model/9/model', 
	resume=False)
