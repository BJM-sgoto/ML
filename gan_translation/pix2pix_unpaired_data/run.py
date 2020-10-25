# reference: https://hardikbansal.github.io/CycleGANBlog/
import numpy as np
import tensorflow.compat.v1 as tf
import os
import random
from PIL import Image

tf.disable_v2_behavior()
tf.reset_default_graph()

IMG_HEIGHT, IMG_WIDTH = 128, 128
LAMBDA = 10

class Model:
	def __init__(self):
		self.kernel_initializer = tf.random_normal_initializer(mean=0.00, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()

	def make_dataset(self, source_folder='./source/', target_folder='./target/'):
		source_images = []
		target_images = []
		for image_file in os.listdir(source_folder):
			source_images.append(source_folder + image_file)
		for image_file in os.listdir(target_folder):
			target_images.append(target_folder + image_file)
		return source_images, target_images
	
	def resnet_block(self, X, depth, name='resnet_block'):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			tX = X 
			tX = tf.layers.conv2d(tX, filters=depth, kernel_size=3, padding='same')
			tX = tf.layers.conv2d(tX, filters=depth, kernel_size=3, padding='same')
			return X + tX

	def generate(self, X, scope, training=False):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			depth = 64
			X = X / 127.5 - 1
			# encode
			X = tf.layers.conv2d(X, filters=depth, kernel_size=7, strides=1, padding='same')
			X = tf.layers.batch_normalization(X, training=training)
			X = tf.nn.leaky_relu(X)
			X = tf.layers.conv2d(X, filters=depth*2, kernel_size=3, strides=2, padding='same')
			X = tf.layers.batch_normalization(X, training=training)
			X = tf.nn.leaky_relu(X)
			X = tf.layers.conv2d(X, filters=depth*4, kernel_size=3, strides=2, padding='same')
			X = tf.layers.batch_normalization(X, training=training)
			X = tf.nn.leaky_relu(X)
			for i in range(5):
				X = self.resnet_block(X, depth=depth*4, name='resnet_block_' + str(i))
				X = tf.layers.batch_normalization(X, training=training)
				X = tf.nn.leaky_relu(X)
			# decode
			X = tf.layers.conv2d_transpose(X, filters=depth*2, kernel_size=3, strides=2, padding='same')
			X = tf.layers.batch_normalization(X, training=training)
			X = tf.nn.leaky_relu(X)
			X = tf.layers.conv2d_transpose(X, filters=depth, kernel_size=3, strides=2, padding='same')
			X = tf.layers.batch_normalization(X, training=training)
			X = tf.nn.leaky_relu(X)
			X = tf.layers.conv2d(X, filters=3, kernel_size=7, strides=1, activation=tf.nn.tanh, padding='same')
			X = (X + 1) * 127.5
			return X
		

	def discriminate(self, X, scope, training=False):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			X = X / 127.5 - 1
			for i in range(4):
				depth = 64*(2**i)
				X = tf.layers.conv2d(X, filters=depth, kernel_size=4, strides=2)
				X = tf.layers.batch_normalization(X, training=training)
				X = tf.nn.relu(X)
			X = tf.layers.conv2d(X, filters=1, kernel_size=4, strides=1, padding='same', activation=tf.nn.sigmoid)
			return X

	def train(self, source_folder='./source/', target_folder='./target/', batch_size=5, num_steps=1000, output_folder='./output/' , model_path='./model/model', resume=False):
		A = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		B = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		gen_B = self.generate(A, scope='generator_AtoB', training=True)
		gen_A = self.generate(B, scope='generator_BtoA', training=True)
		dec_A = self.discriminate(A, scope='discriminator_A', training=True)
		dec_B = self.discriminate(B, scope='discriminator_B', training=True)

		dec_gen_A = self.discriminate(gen_A, scope='discriminator_A', training=True)
		dec_gen_B = self.discriminate(gen_A, scope='discriminator_B', training=True)
		cyc_A = self.generate(gen_B, scope='generator_BtoA', training=True)
		cyc_B = self.generate(gen_A, scope='generator_AtoB', training=True)
		
		# loss
		# discriminator loss
		D_A_loss_1 = tf.reduce_mean(tf.squared_difference(dec_A,1))
		D_B_loss_1 = tf.reduce_mean(tf.squared_difference(dec_B,1))
		
		D_A_loss_2 = tf.reduce_mean(tf.square(dec_gen_A))
		D_B_loss_2 = tf.reduce_mean(tf.square(dec_gen_B))

		d_loss_A = (D_A_loss_1 + D_A_loss_2)/2
		d_loss_B = (D_B_loss_1 + D_B_loss_2)/2
		
		# generator loss
		g_loss_B_1 = tf.reduce_mean(tf.squared_difference(dec_gen_B,1))
		g_loss_A_1 = tf.reduce_mean(tf.squared_difference(dec_gen_A,1))
		
		# cyclic loss
		cyc_loss = tf.reduce_mean(tf.abs(A-cyc_A)/255) + tf.reduce_mean(tf.abs(B-cyc_B)/255)
		
		g_loss_A = 0*g_loss_A_1 + 10*cyc_loss
		g_loss_B = 0*g_loss_B_1 + 10*cyc_loss
		for var in tf.trainable_variables():
			print(var)
		
		# trainable variables
		d_A_vars = tf.trainable_variables(scope='discriminator_A')
		d_B_vars = tf.trainable_variables(scope='discriminator_B')
		g_A_vars = tf.trainable_variables(scope='generator_BtoA')
		g_B_vars = tf.trainable_variables(scope='generator_AtoB')

		# train ops
		d_A_trainer = tf.train.AdamOptimizer().minimize(d_loss_A, var_list=d_A_vars)
		d_B_trainer = tf.train.AdamOptimizer().minimize(d_loss_B, var_list=d_B_vars)
		g_A_trainer = tf.train.AdamOptimizer(1e-4).minimize(g_loss_A, var_list=g_A_vars)
		g_B_trainer = tf.train.AdamOptimizer(1e-4).minimize(g_loss_B, var_list=g_B_vars)

		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())

		source_dataset, target_dataset = self.make_dataset(source_folder, target_folder)
		
		
		count_to_save = 0
		count_to_draw = 0
		for i in range(num_steps):
			source_images = random.sample(source_dataset, batch_size)
			target_images = random.sample(target_dataset, batch_size)

			a = []
			b = []
			for k in range(batch_size):
				source_image = np.float32(Image.open(source_images[k]))
				h, w, _  = source_image.shape
				start_y = np.random.randint(low = 0, high = h - IMG_HEIGHT + 1)
				start_x = np.random.randint(low = 0, high = w - IMG_WIDTH + 1)
				source_image = source_image[start_y: start_y+IMG_HEIGHT, start_x: start_x+IMG_WIDTH]
				a.append(source_image)
				target_image = np.float32(Image.open(target_images[k]))
				h, w, _  = target_image.shape
				start_y = np.random.randint(low = 0, high = h - IMG_HEIGHT + 1)
				start_x = np.random.randint(low = 0, high = w - IMG_WIDTH + 1)
				target_image = target_image[start_y: start_y+IMG_HEIGHT, start_x: start_x+IMG_WIDTH]
				b.append(target_image)
			a = np.float32(a)
			b = np.float32(b)
			# train
			print('---------------\nStep {:06d}'.format(i))
			
			# train on reconstruction loss
			'''
			d_loss_A_val, _ = session.run([d_loss_A, d_A_trainer], feed_dict={A: a, B: b})
			print('d_loss_A: {:06f}'.format(d_loss_A_val))
			
			d_loss_B_val, _ = session.run([d_loss_B, d_B_trainer], feed_dict={A: a, B: b})
			print('d_loss_B: {:06f}'.format(d_loss_B_val))
			'''
			# train on reconstruction loss
			g_loss_A_val, _ = session.run([g_loss_A, g_A_trainer], feed_dict={A: a, B: b})
			print('g_loss_A: {:06f}'.format(g_loss_A_val))
			
			g_loss_B_val, _ = session.run([g_loss_B, g_B_trainer], feed_dict={A: a, B: b})
			print('g_loss_B: {:06f}'.format(g_loss_B_val))
			
			count_to_save+=1
			if count_to_save>=100:
				saver.save(session, model_path)
				count_to_save = 0
				n_image = min(4, batch_size)
				image = np.zeros([4*IMG_HEIGHT+30, n_image*IMG_WIDTH+(n_image-1)*10, 3], dtype=np.float32)
				gen_B_val, gen_A_val = session.run([cyc_A, cyc_B], feed_dict={A: a, B: b})
				for k in range(n_image):
					start_x = k*(IMG_WIDTH+10)
					image[0*IMG_HEIGHT+00: 1*IMG_HEIGHT+00, start_x: start_x+IMG_WIDTH] = a[k]
					image[1*IMG_HEIGHT+10: 2*IMG_HEIGHT+10, start_x: start_x+IMG_WIDTH] = gen_B_val[k]
					image[2*IMG_HEIGHT+20: 3*IMG_HEIGHT+20, start_x: start_x+IMG_WIDTH] = b[k]
					image[3*IMG_HEIGHT+30: 4*IMG_HEIGHT+30, start_x: start_x+IMG_WIDTH] = gen_A_val[k]
				
				image = Image.fromarray(np.uint8(image))
				image.save(output_folder + '{:05d}.png'.format(count_to_draw))
				count_to_draw += 1
		session.close()
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
model = Model()
model.train(
	source_folder = dir_path + 'small_horse/',
	target_folder = dir_path + 'small_zebra/',
	batch_size = 20,
	num_steps = 1000,
	output_folder = dir_path + 'output/',
	model_path = dir_path + 'model/model',
	resume=False)
