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

	def instance_normalization(self, X, scope='instance_normalization'):
		with tf.variable_scope(scope):
			depth = X.get_shape()[-1].value
			mean = tf.reduce_mean(X, axis=[1,2], keep_dims=True)
			stddev = tf.sqrt(tf.reduce_mean(tf.square(X - mean), axis=[1,2], keep_dims=True) + 1e-6)
			X = (X - mean)/ stddev
			new_mean = tf.get_variable(name='new_mean', shape=[depth], dtype=tf.float32)
			new_stddev = tf.nn.softplus(tf.get_variable(name='new_stddev', shape=[depth], dtype=tf.float32))
			X = X  * new_stddev + new_mean
			return X

	def encode(self, X, training=False):
		with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
			Z = X / 127.5 - 1
			Zs = []
			layer_depths = [64,128,256,512]
			for i, layer_depth in enumerate(layer_depths):
				Z = tf.layers.conv2d(Z, filters=layer_depth, kernel_size=4, strides=2, padding='same', kernel_initializer=self.kernel_initializer)
				#Z = tf.layers.batch_normalization(Z, training=training)
				if i!=0:
					Z = self.instance_normalization(Z, scope='instance_normalization_'+str(i))
				Z = tf.nn.leaky_relu(Z)
				Zs.append(Z)
			return Zs

	def decode(self, Zs, training=False):
		with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
			X = Zs.pop()
			layer_depths = [256,128,64]
			for i, layer_depth in enumerate(layer_depths):
				X = tf.layers.conv2d_transpose(X, filters=layer_depth, kernel_size=4, strides=2, padding='same', kernel_initializer=self.kernel_initializer)
				# X = tf.layers.batch_normalization(X, training=training)
				X = self.instance_normalization(X, scope='instance_normalization_' + str(i))
				X = tf.nn.leaky_relu(X)
				X = tf.concat([X, Zs.pop()], axis=3)

				#_, X_height, X_width, _ = X.get_shape().as_list()
				
			X = tf.layers.conv2d_transpose(X, filters=3, kernel_size=4, strides=2, padding='same', activation=tf.nn.tanh)
			X = (X + 1) * 127.5
			return X

	def generate(self, X, scope='generator', training=False):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			X = self.encode(X, training=training)
			X = self.decode(X, training=training)
			return X

	def discriminate(self, X, scope='disciminator', training=False):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			layer_depths = [32,64,128,256]
			Y = X / 127.5 - 1
			for i, layer_depth in enumerate(layer_depths):
				stride = 2
				if i==len(layer_depths)-1:
					stride = 1
				Y = tf.layers.conv2d(Y, filters=layer_depth, kernel_size=4, strides=stride, padding='same', kernel_initializer=self.kernel_initializer)
				if i!=0:
					# Y = tf.layers.batch_normalization(Y, training=training)
					Y = self.instance_normalization(Y, scope='instance_normalization_' + str(i))
				Y = tf.nn.leaky_relu(Y)
			Y = tf.layers.conv2d(Y, filters=1, kernel_size=4, padding='same', activation=tf.nn.sigmoid)
			return Y

	def log_loss(self, target_output, predicted_output):
		loss = - tf.reduce_mean(target_output * tf.log(predicted_output + 1e-6) + (1 - target_output) * tf.log(1 + 1e-6 - predicted_output))
		return loss

	def compute_loss(self, source_images, reconstructed_source_images, target_images, reconstructed_target_images, forward_discriminator_outputs, backward_discriminator_outputs):
		edge = 16
		# number of source and target images
		n_source = tf.shape(source_images)[0]
		n_target = tf.shape(target_images)[0]

		# reconstruction loss
		reconstruction_loss_on_source = tf.reduce_mean(tf.abs((source_images - reconstructed_source_images)/255))
		reconstruction_loss_on_target = tf.reduce_mean(tf.abs((target_images - reconstructed_target_images)/255))

		# adversarial loss on generator
		predicted_outputs = forward_discriminator_outputs[n_target:]
		adversarial_loss_on_forward_generator = tf.reduce_mean(tf.square(1 - predicted_outputs))
		predicted_outputs = backward_discriminator_outputs[n_source:]
		adversarial_loss_on_backward_generator = tf.reduce_mean(tf.square(1 - predicted_outputs))

		# adversarial loss on discriminator
		true_labels = tf.concat([
			tf.ones(shape=[n_target, edge, edge, 1],dtype=tf.float32),
			tf.zeros(shape=[n_source, edge, edge, 1],dtype=tf.float32)], axis=0)
		adversarial_loss_on_forward_discriminator = tf.reduce_mean(tf.square(true_labels - forward_discriminator_outputs))
		true_labels = tf.concat([
			tf.ones(shape=[n_source, edge, edge,1], dtype=tf.float32),
			tf.zeros(shape=[n_target, edge, edge,1], dtype=tf.float32)], axis=0)
		adversarial_loss_on_backward_discriminator = tf.reduce_mean(tf.square(true_labels - backward_discriminator_outputs))
		return reconstruction_loss_on_source, reconstruction_loss_on_target, adversarial_loss_on_forward_generator, adversarial_loss_on_backward_generator, adversarial_loss_on_forward_discriminator, adversarial_loss_on_backward_discriminator

	def train(self, source_folder='./source/', target_folder='./target/', batch_size=5, num_steps=1000, output_folder='./output/' , model_path='./model/model', resume=False):
		X1 = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		X2 = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		FW_X1 = self.generate(X1, scope='forward_generator', training=True)
		BW_X1 = self.generate(FW_X1, scope='backward_generator', training=True)
		BW_X2 = self.generate(X2, scope='backward_generator', training=True)
		FW_X2 = self.generate(BW_X2, scope='forward_generator', training=True)

		CX_1 = tf.concat([X2, FW_X1], axis=0)
		FW_Y = self.discriminate(CX_1, scope='forward_discriminator', training=True)
		CX_2 = tf.concat([X1, BW_X2], axis=0)
		BW_Y = self.discriminate(CX_2, scope='backward_discriminator', training=True)

		reconstruction_loss_on_source, reconstruction_loss_on_target, adversarial_loss_on_forward_generator, adversarial_loss_on_backward_generator, adversarial_loss_on_forward_discriminator, adversarial_loss_on_backward_discriminator = self.compute_loss(X1, BW_X1, X2, FW_X2, FW_Y, BW_Y)
		
		# train ops
		# train ops of generators on based on reconstruction loss
		generator_on_source_train_op = tf.train.AdamOptimizer(1e-4).minimize(LAMBDA*reconstruction_loss_on_source, var_list=tf.trainable_variables('forward_generator') + tf.trainable_variables('backward_generator'))
		generator_on_target_train_op = tf.train.AdamOptimizer(1e-4).minimize(LAMBDA*reconstruction_loss_on_target, var_list=tf.trainable_variables('forward_generator') + tf.trainable_variables('backward_generator'))
		
		# train ops of generators on based on adversarial loss
		forward_generator_train_op = tf.train.AdamOptimizer(1e-4).minimize(adversarial_loss_on_forward_generator, var_list=tf.trainable_variables('forward_generator'))
		backward_generator_train_op = tf.train.AdamOptimizer(1e-4).minimize(adversarial_loss_on_backward_generator, var_list=tf.trainable_variables('backward_generator'))
		# train ops of discriminator on based on adversarial loss
		forward_discriminator_train_op = tf.train.AdamOptimizer(1e-4).minimize(adversarial_loss_on_forward_discriminator, var_list=tf.trainable_variables('forward_discriminator'))
		backward_discriminator_train_op = tf.train.AdamOptimizer(1e-4).minimize(adversarial_loss_on_backward_discriminator, var_list=tf.trainable_variables('backward_discriminator'))
		
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

			x1 = []
			x2 = []
			for k in range(batch_size):
				source_image = np.float32(Image.open(source_images[k]))
				h, w, _  = source_image.shape
				start_y = np.random.randint(low = 0, high = h - IMG_HEIGHT + 1)
				start_x = np.random.randint(low = 0, high = w - IMG_WIDTH + 1)
				source_image = source_image[start_y: start_y+IMG_HEIGHT, start_x: start_x+IMG_WIDTH]
				x1.append(source_image)
				target_image = np.float32(Image.open(target_images[k]))
				h, w, _  = target_image.shape
				start_y = np.random.randint(low = 0, high = h - IMG_HEIGHT + 1)
				start_x = np.random.randint(low = 0, high = w - IMG_WIDTH + 1)
				target_image = target_image[start_y: start_y+IMG_HEIGHT, start_x: start_x+IMG_WIDTH]
				x2.append(target_image)
			x1 = np.float32(x1)
			x2 = np.float32(x2)
			# train
			print('---------------\nStep {:06d}'.format(i))
			# train on reconstruction loss
			reconstruction_loss_on_source_val, _ = session.run([reconstruction_loss_on_source, generator_on_source_train_op], feed_dict={X1: x1})
			print('Reconstruction Loss on Source: {:06f}'.format(reconstruction_loss_on_source_val))
			
			reconstruction_loss_on_target_val, _ = session.run([reconstruction_loss_on_target, generator_on_target_train_op], feed_dict={X2: x2})
			print('Reconstruction Loss on Target: {:06f}'.format(reconstruction_loss_on_target_val))
			
			# train generator on adversarial loss
			adversarial_loss_on_forward_generator_val, _ = session.run([adversarial_loss_on_forward_generator, forward_generator_train_op], feed_dict={X1: x1, X2:x2})
			print('Adversarial Loss Forward Generator {:06f}'.format(adversarial_loss_on_forward_generator_val))

			adversarial_loss_on_backward_generator_val, _ = session.run([adversarial_loss_on_backward_generator, backward_generator_train_op], feed_dict={X1: x1, X2: x2})
			print('Adversarial Loss Backward Generator {:06f}'.format(adversarial_loss_on_backward_generator_val))

			# train discriminator on adversarial loss		
			adversarial_loss_on_forward_discriminator_val, _ = session.run([adversarial_loss_on_forward_discriminator, forward_discriminator_train_op], feed_dict={X1: x1, X2: x2})
			print('Adversarial Loss Forward Discriminator {:06f}'.format(adversarial_loss_on_forward_discriminator_val))

			adversarial_loss_on_backward_discriminator_val, _ = session.run([adversarial_loss_on_backward_discriminator, backward_discriminator_train_op], feed_dict={X1: x1, X2: x2})
			print('Adversarial Loss Backward Discriminator {:06f}'.format(adversarial_loss_on_backward_discriminator_val))
			
			count_to_save+=1
			if count_to_save>=100:
				saver.save(session, model_path)
				count_to_save = 0
				n_image = min(5, batch_size)
				image = np.zeros([5*IMG_HEIGHT+40, n_image*IMG_WIDTH+(n_image-1)*10, 3], dtype=np.float32)
				reconstructed_soure_images, fake_target_images, reconstructed_target_images = session.run([BW_X1, FW_X1, FW_X2], feed_dict={X1: x1, X2: x2})
				for k in range(n_image):
					start_x = k*(IMG_WIDTH+10)
					image[0*IMG_HEIGHT+00: 1*IMG_HEIGHT+00, start_x: start_x+IMG_WIDTH] = x1[k]
					image[1*IMG_HEIGHT+10: 2*IMG_HEIGHT+10, start_x: start_x+IMG_WIDTH] = reconstructed_soure_images[k]
					image[2*IMG_HEIGHT+20: 3*IMG_HEIGHT+20, start_x: start_x+IMG_WIDTH] = fake_target_images[k]
					image[3*IMG_HEIGHT+30: 4*IMG_HEIGHT+30, start_x: start_x+IMG_WIDTH] = x2[k]
					image[4*IMG_HEIGHT+40: 5*IMG_HEIGHT+40, start_x: start_x+IMG_WIDTH] = reconstructed_target_images[k]
				
				image = Image.fromarray(np.uint8(image))
				image.save(output_folder + '{:05d}.png'.format(count_to_draw))
				count_to_draw += 1
		session.close()

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
model = Model()
model.train(
	source_folder=dir_path + 'small_horse/',
	target_folder=dir_path + 'small_zebra/',
	batch_size=10,
	num_steps=5000,
	output_folder=dir_path + 'output/',
	model_path=dir_path + 'model/model',
	resume=False)