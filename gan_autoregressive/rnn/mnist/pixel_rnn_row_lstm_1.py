import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
import os

tf.disable_v2_behavior()
tf.reset_default_graph()

class Ops:
	weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
	bias_initializer = tf.zeros_initializer()
	
	@staticmethod 
	def make_weight1d(in_channel, out_channel, kernel_size, kernel_format='L,R,C'):
		# default kernel is fullly filled kernel: common kernel
		left, right = kernel_size
		weight = tf.get_variable('weight', shape=[left+right+1, in_channel, out_channel], dtype=tf.float32, initializer=Ops.weight_initializer)
		mask = np.zeros([left+right+1, in_channel, out_channel], dtype=np.float32)
		kernel_elements = kernel_format.split(',')
		if 'L' in kernel_elements:
			mask[:left] = 1.0
		if 'R' in kernel_elements:
			mask[left+1:] = 1.0
		if 'C' in kernel_elements:
			mask[left] = 1.0
		weight = weight * mask
		return weight
		
	@staticmethod
	def make_weight2d(in_channel, out_channel, kernel_size, kernel_format='N,S,E,W,NW,NE,SE,SW,C'):
		
		# default kernel is fullly filled kernel: common kernel
		left, top, right, bottom = kernel_size
		weight = tf.get_variable('weight', shape=[top+bottom+1, left+right+1, in_channel, out_channel], dtype=tf.float32, initializer=Ops.weight_initializer)
		mask = np.zeros([top+bottom+1, left+right+1, in_channel, out_channel], dtype=np.float32)
		kernel_elements = kernel_format.split(',')
		if 'N' in kernel_elements:
			mask[:top, left] = 1.0
		if 'S' in kernel_elements:
			mask[top+1:, left] = 1.0
		if 'E' in kernel_elements:
			mask[top, left+1:] = 1.0
		if 'W' in kernel_elements:
			mask[top, :left] = 1.0
		if 'NW' in kernel_elements:
			mask[:top, :left] = 1.0
		if 'NE' in kernel_elements:
			mask[:top, left+1:] = 1.0
		if 'SW' in kernel_elements:
			mask[top+1:, :left] = 1.0
		if 'SE' in kernel_elements:
			mask[top+1:, left+1:] = 1.0
		if 'C' in kernel_elements:
			mask[top, left] = 1.0
		weight = weight * mask
		return weight
	
	@staticmethod
	def conv1d(x, filters, kernel_size, mask_format, use_bias=False, scope='conv1d'):
		with tf.variable_scope(scope):
			left, right = kernel_size
			# weight
			weight = Ops.make_weight1d(x.get_shape()[-1].value, filters, kernel_size, mask_format)		
			# pad => pad 0 because we do not have info about anything around
			paddings = [[0,0],[left, right],[0,0]]
			x = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)
			x = tf.nn.conv1d(x, filters=weight, stride=1, padding='VALID')
			if use_bias:
				# bias
				bias = tf.get_variable('bias', shape=[filters], dtype=tf.float32, initializer=Ops.bias_initializer)
				x = x + bias
			return x
			
	@staticmethod
	def conv2d(x, filters, kernel_size, mask_format, use_bias=False, scope='conv2d'):
		with tf.variable_scope(scope):
			# size 
			left, top, right, bottom = kernel_size			
			# weight
			weight = Ops.make_weight2d(x.get_shape()[-1].value, filters, kernel_size, mask_format)			
			# pad => pad 0 because we do not have info about anything around
			paddings = [[0,0],[top, bottom],[left, right],[0,0]]
			x = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)
			x = tf.nn.conv2d(x, filters=weight, strides=(1,1), padding='VALID')
			
			if use_bias:
				# bias
				bias = tf.get_variable('bias', shape=[filters], dtype=tf.float32, initializer=Ops.bias_initializer)
				x = x + bias
			return x
	
class RowLSTMCell(tf.nn.rnn_cell.RNNCell):
	def __init__(self, hidden_dims, length):
		self._length = length
		self._hidden_dims = hidden_dims
		self._num_units = length * hidden_dims
		self._state_size = (self._num_units, self._num_units)
		self._output_size = self._num_units
		
	@property
	def state_size(self):
		return self._state_size

	@property
	def output_size(self):
		return self._output_size
	
	def __call__(self, inputs, states, scope="row_lstm_cell"):
		#c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
		#h_prev = tf.slice(state, [0, self._num_units], [-1, self._num_units]) # [batch, height * hidden_dims]

		c_prev = states[0]
		h_prev = states[1]
		with tf.variable_scope(scope):
			# reform inputs
			_, input_depth = inputs.get_shape().as_list()
			inputs = tf.reshape(inputs, [-1, self._length, int(input_depth//self._length)])
			inputs = tf.layers.dense(inputs, units=self._hidden_dims, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
			inputs = tf.reshape(inputs, [-1, self._num_units])			
			
			h_prev = tf.reshape(h_prev, [-1, self._length, self._hidden_dims])
			# make f
			h_prev_to_f = Ops.conv1d(h_prev, self._hidden_dims, kernel_size=[1,1], mask_format='L,C,R', use_bias=True, scope='h_prev_to_f')
			f = tf.nn.sigmoid(inputs + tf.reshape(h_prev_to_f, [-1, self._num_units]))
			# make i
			h_prev_to_i = Ops.conv1d(h_prev, self._hidden_dims, kernel_size=[1,1], mask_format='L,C,R', use_bias=True, scope='h_prev_to_i')
			i = tf.nn.sigmoid(inputs + tf.reshape(h_prev_to_i, [-1, self._num_units]))
			# make c
			h_prev_to_c_ = Ops.conv1d(h_prev, self._hidden_dims, kernel_size=[1,1], mask_format='L,C,R', use_bias=True, scope='h_prev_to_c_')
			c_ = tf.nn.tanh(inputs + tf.reshape(h_prev_to_c_, [-1, self._num_units]))
			# make o
			h_prev_to_o = Ops.conv1d(h_prev, self._hidden_dims, kernel_size=[1,1], mask_format='L,C,R', use_bias=True, scope='h_prev_to_o')
			o = tf.nn.sigmoid(inputs + tf.reshape(h_prev_to_o, [-1, self._num_units]))
			# output and states
			c = tf.nn.sigmoid(f*c_prev + i*c_)
			h = tf.nn.tanh(c) * o
		
		return h, (c, h)
		
IMG_HEIGHT, IMG_WIDTH = 28, 28
HIDDEN_DIM = 64
QUANTIZE_LEVELS = 4
class Model:
	def __init__(self):
		self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()
	
	def make_dataset(self, image_folder='./image/'):
		dataset = []
		count = 0
		classes = ['0']
		for sub_folder in os.listdir(image_folder):
			if sub_folder not in classes:
				continue
			sub_path = image_folder + sub_folder + '/'
			for image_name in os.listdir(sub_path):
				image = np.int32(np.int32(Image.open(sub_path + image_name))/(256/QUANTIZE_LEVELS))
				extended_image = np.float32(np.zeros([IMG_HEIGHT, IMG_WIDTH, QUANTIZE_LEVELS]))
				for i in range(IMG_HEIGHT):
					for j in range(IMG_WIDTH):
						extended_image[i,j,image[i,j]] = 1
				dataset.append(extended_image)
				count+=1
				if count%1000==0:
					print(count)
		dataset = np.float32(dataset)
		return dataset
	
	def shuffle_dataset(self, dataset):
		np.random.shuffle(dataset)
	
	def row_lstm(self, x, scope='row_lstm'):
		with tf.variable_scope(scope):
			_, height, width, channel = x.get_shape().as_list()
			rnn_inputs = tf.reshape(x, [-1, height, width*channel]) # batch * width * feature
			rnn_cell = RowLSTMCell(HIDDEN_DIM, width)
			output, states = tf.nn.dynamic_rnn(
				rnn_cell,
				rnn_inputs,
				dtype=tf.float32,
				time_major=False)
			depth = output.get_shape()[-1].value
			output = tf.reshape(output, [-1, width, height, int(depth//height)])
			output = tf.layers.dense(output, units=QUANTIZE_LEVELS, use_bias=False, kernel_initializer=self.weight_initializer)
			output = tf.nn.softmax(output, axis=3)
			return output
	
	def auto_regress(self, x):
		x = tf.layers.dense(x, units=HIDDEN_DIM)
		x = Ops.conv2d(x, filters=HIDDEN_DIM, kernel_size=[1,1,1,0], mask_format='N,NW,NE', use_bias=False, scope='conv2d_0')
		x = tf.nn.leaky_relu(x, 0.2)
		
		for i in range(1,9):
			x = Ops.conv2d(x, filters=HIDDEN_DIM, kernel_size=[1,1,1,0], mask_format='N,NW,NE,W,E,C', use_bias=False, scope='conv2d_' + str(i))
			x = tf.nn.leaky_relu(x, 0.2)
		
		x = self.row_lstm(x)
		return x
	
	def compute_cost(self, target_output, predicted_output):
		cost = tf.reduce_mean(tf.square(target_output - predicted_output))
		return cost
	
	def train(self, num_epoch=50, batch_size=50, image_folder='./image/',output_folder='./output/', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, QUANTIZE_LEVELS])
		Y = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, QUANTIZE_LEVELS])
		PY = self.auto_regress(X)
		cost = self.compute_cost(Y, PY)
		train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
		saver = tf.train.Saver()
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
				y = dataset[j: end_j]
				py_val, cost_val,_ = session.run([PY, cost, train_op], feed_dict={X: x, Y: y})
				print('Epoch {:03d}, Progress {:06d}, Loss {:06f}'.format(i, j, cost_val))
			saver.save(session, model_path)
		session.close()
	
	def generate_image(self, model_path='./model/model', num_samples_on_edge=10, output_file='./output.png'):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, QUANTIZE_LEVELS])
		Y = self.auto_regress(X)
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)
		
		n = num_samples_on_edge
		n_samples = n * n
		# make sub images
		images = np.zeros([n_samples, IMG_HEIGHT, IMG_WIDTH, QUANTIZE_LEVELS], dtype=np.float32)
		for i in range(IMG_HEIGHT):
			new_images = session.run(Y, feed_dict={X: images})
			for j in range(IMG_WIDTH):
				for k in range(n_samples):
					prob = new_images[k, i, j]
					choice = np.random.choice(QUANTIZE_LEVELS, p=prob)
					images[k, i, j] = 0
					images[k, i, j, choice] = 1
		images = np.argmax(images,axis=3)*(256/QUANTIZE_LEVELS)
		# paste sub images to image
		image = np.zeros([n*IMG_HEIGHT+(n-1)*10,n*IMG_HEIGHT+(n-1)*10], dtype=np.float32)
		for i in range(n):
			for j in range(n):
				image[i*(IMG_HEIGHT+10):i*(IMG_HEIGHT+10)+IMG_HEIGHT, j*(IMG_WIDTH+10):j*(IMG_WIDTH+10)+IMG_WIDTH]=images[i*n+j]
		image = Image.fromarray(np.uint8(image))
		image.save(output_file)
		
model = Model()
'''
model.train(
	num_epoch=200,
	batch_size=50,
	image_folder='./train/',
	output_folder='./output/',
	model_path='./model/model_row_1',
	resume=False)
'''
model.generate_image(
	model_path='./model/model_row_1',
	num_samples_on_edge=10,
	output_file='./output.png')
