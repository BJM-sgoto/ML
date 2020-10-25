# small test on fashion mnist
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random

RANDOM_SEED = 1234
tf.disable_v2_behavior()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

IMG_WIDTH = 28
IMG_HEIGHT = 28
Z_DIM = 100

class Model:
	def __init__(self):
		self.reuse_generator = False
		self.reuse_discriminator = False
		
	def load_dataset(self, image_file='./image.npy', label_file='./labels.npy'):
		images = np.load(image_file)
		labels = np.load(label_file)
		dataset = list(zip(images, labels))
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def generate(self, z, c, training=False):
		with tf.variable_scope('generator', reuse=self.reuse_generator):
			x = tf.concat([z, c], axis=1)
			x = tf.layers.dense(x, units= 128 * 7 * 7)
			x = tf.nn.leaky_relu(x, alpha=0.2)
			x = tf.reshape(x, [-1, 7, 7, 128])
			# upsample to 14X14
			x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, 0.2)
			# upsample to 28X28
			x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, 0.2)
			# generate
			x = tf.layers.conv2d(x, filters=1, kernel_size=7, activation=tf.nn.tanh, padding='same')
			x = (x + 1) * 127.5
			self.reuse_generator = True
			return x
	
	def discriminate(self, x, c, training=False):
		with tf.variable_scope('discriminator', reuse=self.reuse_discriminator):
			x = x / 127.5 - 1
			
			# downsample
			x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, alpha=0.2)
			
			# downsample
			x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same')
			x = tf.nn.leaky_relu(x, alpha=0.2)
			
			# discriminate
			x = tf.layers.flatten(x)
			x = tf.layers.dropout(x, 0.4, training=training)
			x = tf.layers.dense(x, units=10)
			x = tf.concat([x, c], axis=1)
			x = tf.layers.dense(x, units = 1, activation=tf.nn.sigmoid)
			x = tf.squeeze(x, 1)
			
			self.reuse_discriminator = True
			return x
	
	def compute_cost(self, predicted_outputs):
		batch_size = tf.shape(predicted_outputs)[0]
		half_batch = tf.cast(batch_size/2, dtype=tf.int32)
		fake_target_outputs = tf.random.uniform(minval=0.80, maxval=0.99, dtype=tf.float32, shape=[half_batch])
		generator_cost = - tf.reduce_mean(fake_target_outputs * tf.log(predicted_outputs[half_batch:] + 1e-6) + (1 - fake_target_outputs) * tf.log(1 - predicted_outputs[half_batch:] + 1e-6))
		
		true_target_outputs = tf.concat(
		[tf.random.uniform(minval=0.80, maxval=0.99, dtype=tf.float32, shape=[half_batch]),
		tf.random.uniform(minval=0.01, maxval=0.20, dtype=tf.float32, shape=[half_batch])], axis=0)
		discriminator_cost = - tf.reduce_mean(true_target_outputs * tf.log(predicted_outputs + 1e-6) + (1 -  true_target_outputs) * tf.log(1 - predicted_outputs + 1e-6))
		return generator_cost, discriminator_cost
	
	def train(self, num_epochs=100, batch_size=128, image_file='./images.npy', label_file='./labels.npy', model_path='./fashion_mnist/model', resume=False):
		Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
		FC = tf.placeholder(tf.float32, shape=[None, 10])
		RX = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 1])
		FX = self.generate(Z, FC, training=True)
		X = tf.concat([RX, FX], axis=0)
		RC = tf.placeholder(tf.float32, shape=[None, 10])
		C = tf.concat([RC, FC], axis=0)
		PY = self.discriminate(X, C, training=True)
		
		
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
			
		dataset = self.load_dataset(image_file=image_file, label_file=label_file)
		num_data = len(dataset)
		count_to_draw = 0
		count_to_save = 0
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j + batch_size)
				rx = []
				rc = []
				for k in range(j, end_j):
					image = dataset[k][0]
					rx.append(np.reshape(image, [28, 28, 1]))
					tc = np.zeros([10], dtype=np.float32)
					tc[dataset[k][1]] = 1
					rc.append(tc)
				rx = np.float32(rx)
				z = np.random.uniform(low=-1, high=1, size=[end_j - j, Z_DIM])
				rc = np.float32(rc)
				tc = np.random.randint(size=[end_j-j], low=0, high=10)
				fc = np.zeros([end_j-j, 10], dtype=np.float32)
				fc[range(end_j-j), tc] = 1
				# train generator
				gen_loss_val, _ = session.run([generator_cost, generator_train_op], feed_dict={Z: z, RX: rx, RC: rc, FC: fc})
				# train discriminator
				dis_loss_val, _ = session.run([discriminator_cost, discriminator_train_op], feed_dict={Z: z, RX: rx, RC: rc, FC: fc})
				count_to_draw += 1
				print('Count {:03d}, Epoch {:02d}, Progress {:05d}, GL {:06f}, DL {:06f}'.format(count_to_draw, i, j, gen_loss_val, dis_loss_val))
				if count_to_draw>=50:
					count_to_draw = 0
					z = np.random.uniform(low=-1, high=1, size=[100, Z_DIM])
					tc = np.repeat(range(10), 10)
					fc = np.zeros([100, 10], dtype=np.float32)
					fc[range(100), tc] = 1
					fx = session.run(FX, feed_dict={Z: z, FC: fc})
					image = np.ones([IMG_HEIGHT * 10 + 10 * 9, IMG_WIDTH * 10 + 10 * 9, 1]) * 255
					for m in range(10):
						for n in range(10):
							image[m*(IMG_HEIGHT + 10): m*(IMG_HEIGHT + 10) + IMG_HEIGHT, n*(IMG_WIDTH + 10): n*(IMG_WIDTH + 10) + IMG_WIDTH] = fx[m*10+n]
					cv2.imwrite('./fashionmnist/test_{:06d}.jpg'.format(count_to_save), image)
					count_to_save += 1
			saver.save(session, model_path)
		session.close()
		
		
model = Model()
model.train(
	image_file='./fashionmnist/train_images.npy', 
	label_file='./fashionmnist/train_labels.npy',
	model_path='./fashionmnist/model', 
	resume=True)