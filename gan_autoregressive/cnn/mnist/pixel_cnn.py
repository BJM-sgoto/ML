import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import os

IMG_HEIGHT = 28
IMG_WIDTH = 28
DIM = 64
QUANTIZE_LEVEL = 4
LOAD_TO_MEMORY = True

tf.disable_v2_behavior()
tf.reset_default_graph()

class Model:
	def __init__(self):
		self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()
	
	def make_dataset(self, image_folder='./image/'):
		dataset = []
		count = 0
		# classes = ['0','1','2','3','4','5','6','7','8','9']
		classes = ['0']
		for sub_folder in os.listdir(image_folder):
			if sub_folder not in classes:
				continue
			sub_path = image_folder + sub_folder + '/'
			for image_name in os.listdir(sub_path):
				image = np.float32(Image.open(sub_path + image_name))
				dataset.append(image)
				count+=1
				if count%1000==0:
					print(count)
		dataset = np.float32(dataset)//(256/QUANTIZE_LEVEL)
		dataset = np.uint8(dataset)
		return dataset
	
	def shuffle_dataset(self, dataset):
		np.random.shuffle(dataset)
	
	# kernel_size = [left, top, right, bottom]
	def conv2d(self, x, filters, kernel_size, use_bias=False, mask_type='B', scope='conv2d'):
		with tf.variable_scope(scope):
			n_in_channel = x.get_shape()[-1].value
			# size 
			left, top, right, bottom = kernel_size
			
			# weight
			weight = tf.get_variable('weight', shape=[top+bottom+1, left+right+1, n_in_channel, filters], dtype=tf.float32, initializer=self.weight_initializer)
			
			# mask
			mask = np.ones([top + bottom + 1, left + right + 1, n_in_channel, filters], dtype=np.float32)
			mask[top+1:] = 0.0
			mask[top, left+1:] = 0.0
			if mask_type=='A':
				mask[top, left] = 0.0
			weight = weight * mask
			
			# pad => pad 0 because we do not have info about anything around
			paddings = [[0,0],[top, bottom],[left, right],[0,0]]
			x = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)
			x = tf.nn.conv2d(x, filters=weight, strides=(1,1), padding='VALID')
			
			if use_bias:
				# bias
				bias = tf.get_variable('bias', shape=[filters], dtype=tf.float32, initializer=self.bias_initializer)
				x = x + bias
				
			return x
	
	def generate(self, x, training=False, scope='generator'):
		with tf.variable_scope(scope):
			dictionary = np.float32(np.eye(QUANTIZE_LEVEL))
			x = tf.cast(x, dtype=tf.int32)
			x = tf.expand_dims(x, axis=3)
			x = tf.gather_nd(dictionary, x)
			y = x
			# vertical_stack
			y = self.conv2d(y, filters=DIM, kernel_size=[3,3,3,3], use_bias=True, mask_type='B', scope='conv2d_vertical_stack')
			y = self.conv2d(y, filters=DIM, kernel_size=[3,0,0,0], use_bias=True, mask_type='A', scope='conv2d_horizontal_stack')
			# => RField = [6,3,2,0]
			# => enlarge image to the right side
			for i in range(4):
				y =  self.conv2d(y, filters=DIM, kernel_size=[1,2,2,0], use_bias=True, mask_type='B', scope='conv2d_enlarge_' + str(i))
				y = tf.nn.leaky_relu(y, 0.2)
				y = tf.layers.batch_normalization(y, training=training)				
			# => RField = [10,11,10,0]
			y = tf.layers.dense(y, units=QUANTIZE_LEVEL, use_bias=False)
			y = tf.nn.softmax(y, axis=-1)
			cost = tf.reduce_mean(tf.square(x - y))
			return y, cost
	
	def train(self, num_epoch=50, batch_size=50, image_folder='./image/',output_folder='./output/', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.uint8, shape=[None, IMG_HEIGHT, IMG_WIDTH])
		Y, cost = self.generate(X, training=True, scope='generator')
		saver = tf.train.Saver()
		train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		dataset = self.make_dataset(image_folder=image_folder)
		num_data = len(dataset)
		for i in range(num_epoch):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(j+batch_size, num_data)
				x = dataset[j: end_j]
				cost_val,_ = session.run([cost, train_op], feed_dict={X: x})
				print('Epoch {:03d}, Progress {:06d}, Loss {:06f}'.format(i, j, cost_val))
			image = self.generate_image(session, X, Y)
			image.save(output_folder + '{:06d}.png'.format(i))
			exit()
			saver.save(session, model_path)
		session.close()
		
	def generate_image(self, session, X, Y):
		n = 10
		num_samples = n*n
		choices = np.arange(QUANTIZE_LEVEL)
		images = np.zeros([num_samples, IMG_HEIGHT, IMG_WIDTH], dtype=np.uint8)
		for i in range(IMG_HEIGHT):
			for j in range(IMG_WIDTH):
				y = session.run(Y, feed_dict={X: images})
				for k in range(num_samples):
					c = np.random.choice(choices, p=y[k,i,j])
					images[k,i,j] = c*(256/QUANTIZE_LEVEL)
		print('Max value', np.max(images))
		image = np.zeros([n*IMG_HEIGHT + (n-1)*10, n*IMG_HEIGHT + (n-1)*10], dtype=np.uint8)
		for i in range(n):
			for j in range(n):
				image[i*(IMG_HEIGHT + 10):i*(IMG_HEIGHT + 10)+ IMG_HEIGHT, j*(IMG_HEIGHT + 10):j*(IMG_HEIGHT + 10)+ IMG_HEIGHT] = images[i*n+j]
		image = Image.fromarray(image)
		return image

model = Model()
model.train(
	num_epoch=50,
	batch_size=50,
	image_folder='./train/',
	output_folder='./output/',
	model_path='./model/model', 
	resume=False)