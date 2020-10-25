# style dataset https://www.kaggle.com/ikarus777/best-artworks-of-all-time
# https://github.com/ftokarev/tf-adain
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
import os
import h5py
import random

IMG_HEIGHT, IMG_WIDTH = 256, 256
STYLE_LOSS_WEIGHT = 0.1
LEARNING_RATE = 5e-4
VGG = 'VGG16'
tf.disable_v2_behavior()
tf.reset_default_graph()

class Model:
	def __init__(self):
		self.kernel_initializer = tf.random_normal_initializer(mean=0.00, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()
	
	def make_dataset(self, content_folder='./content/', style_folder='./style/'):
		contents = []
		styles = []
		for image_file in os.listdir(content_folder):
			contents.append(content_folder + image_file)
		for image_file in os.listdir(style_folder):
			styles.append(style_folder + image_file)
		return {'content': contents, 'style': styles}
	
	def sample_data(self, dataset, num_content, num_style):
		contents = random.sample(dataset['content'], num_content)
		styles = random.sample(dataset['style'], num_style)
		return contents, styles
	
	def preprocess_input(self, images):
		# image in format RGB
		return images - np.float32([123.68, 116.779, 103.939])
	
	def vgg_conv2d(self, x, weights, biases):
		x = tf.nn.conv2d(x, filters=weights, strides=(1,1), padding='SAME') + biases
		x = tf.nn.relu(x)
		return x
	
	def vgg_pool(self, x):
		x = tf.layers.max_pooling2d(
			x,
			pool_size=(2,2),
			strides=(2,2),
			padding='SAME')
		return x
	
	def vgg16(self, x, vgg_file='vgg16_notop.h5'):
		f = h5py.File(vgg_file, 'r')
		x = self.preprocess_input(x)
		x = tf.reverse(x, axis=[-1])
		# block1
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block1_conv1']['block1_conv1_W_1:0']),
			biases=np.float32(f['block1_conv1']['block1_conv1_b_1:0']))
		x_out1 = x
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block1_conv2']['block1_conv2_W_1:0']),
			biases=np.float32(f['block1_conv2']['block1_conv2_b_1:0']))
		x = self.vgg_pool(x)
		# block2
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block2_conv1']['block2_conv1_W_1:0']),
			biases=np.float32(f['block2_conv1']['block2_conv1_b_1:0']))
		x_out2 = x
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block2_conv2']['block2_conv2_W_1:0']),
			biases=np.float32(f['block2_conv2']['block2_conv2_b_1:0']))
		x = self.vgg_pool(x)
		# block3
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block3_conv1']['block3_conv1_W_1:0']),
			biases=np.float32(f['block3_conv1']['block3_conv1_b_1:0']))
		x_out3 = x
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block3_conv2']['block3_conv2_W_1:0']),
			biases=np.float32(f['block3_conv2']['block3_conv2_b_1:0']))
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block3_conv3']['block3_conv3_W_1:0']),
			biases=np.float32(f['block3_conv3']['block3_conv3_b_1:0']))
		x = self.vgg_pool(x)
		# block4
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block4_conv1']['block4_conv1_W_1:0']),
			biases=np.float32(f['block4_conv1']['block4_conv1_b_1:0']))
		x_out4 = x
		f.close()
		return x_out1, x_out2, x_out3, x_out4
	
	def vgg19(self, x, vgg_file='vgg19_notop.h5'):
		f = h5py.File(vgg_file, 'r')
		x = self.preprocess_input(x)
		x = tf.reverse(x, axis=[-1])
		# block1
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block1_conv1']['block1_conv1_W_1:0']),
			biases=np.float32(f['block1_conv1']['block1_conv1_b_1:0']))
		x_out1 = x
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block1_conv2']['block1_conv2_W_1:0']),
			biases=np.float32(f['block1_conv2']['block1_conv2_b_1:0']))
		x = self.vgg_pool(x)
		# block2
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block2_conv1']['block2_conv1_W_1:0']),
			biases=np.float32(f['block2_conv1']['block2_conv1_b_1:0']))
		x_out2 = x
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block2_conv2']['block2_conv2_W_1:0']),
			biases=np.float32(f['block2_conv2']['block2_conv2_b_1:0']))
		x = self.vgg_pool(x)
		# block3
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block3_conv1']['block3_conv1_W_1:0']),
			biases=np.float32(f['block3_conv1']['block3_conv1_b_1:0']))
		x_out3 = x
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block3_conv2']['block3_conv2_W_1:0']),
			biases=np.float32(f['block3_conv2']['block3_conv2_b_1:0']))
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block3_conv3']['block3_conv3_W_1:0']),
			biases=np.float32(f['block3_conv3']['block3_conv3_b_1:0']))
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block3_conv4']['block3_conv4_W_1:0']),
			biases=np.float32(f['block3_conv4']['block3_conv4_b_1:0']))
		x = self.vgg_pool(x)
		# block4
		x = self.vgg_conv2d(
			x, 
			weights=np.float32(f['block4_conv1']['block4_conv1_W_1:0']),
			biases=np.float32(f['block4_conv1']['block4_conv1_b_1:0']))
		x_out4 = x
		
		f.close()
		return x_out1, x_out2, x_out3, x_out4
	
	def decode(self, x):
		blocks = [[256], [256,256,256,128], [128,64], [64,3]]
		if VGG=='VGG16':
			blocks = [[256], [256,256,128], [128,64], [64,3]]
		for i, block in enumerate(blocks):
			for layer_depth in block:
				x = tf.layers.conv2d(
					x,
					filters=layer_depth,
					kernel_size=3,
					strides=1,
					padding='same',
					kernel_initializer=self.kernel_initializer,
					bias_initializer=self.bias_initializer,
					activation=tf.nn.relu)
			if i!=len(blocks)-1:
				print('---------------Reshape---------------')
				_, h, w, _ = x.get_shape().as_list()
				x = tf.image.resize(x, (h*2, w*2))
		x = tf.clip_by_value(x, 0, 255)
		print(x)
		return x
	
	# contents and styles
	# inputing n_c contents and n_s styles will produce n_c * n_s images
	def synthesize(self, c, s, vgg_file='vgg_notop.h5'):
		n_c = tf.shape(c)[0]
		n_s = tf.shape(s)[0]
		a = tf.concat([c,s], axis=0)
		if VGG=='VGG16':
			a_out1, a_out2, a_out3, a_out4 = self.vgg16(a, vgg_file=vgg_file)
		else:
			a_out1, a_out2, a_out3, a_out4 = self.vgg19(a, vgg_file=vgg_file)
		c_out1, c_out2, c_out3, c_out4 = a_out1[:n_c], a_out2[:n_c], a_out3[:n_c], a_out4[:n_c]
		s_out1, s_out2, s_out3, s_out4 = a_out1[n_c:], a_out2[n_c:], a_out3[n_c:], a_out4[n_c:]
		
		# duplicate contents [c1,c2,c3]=>[c1,c2,c3,c1,c2,c3,...]
		c_out1 = tf.tile(c_out1, [n_s, 1, 1, 1])
		c_out2 = tf.tile(c_out2, [n_s, 1, 1, 1])
		c_out3 = tf.tile(c_out3, [n_s, 1, 1, 1])
		c_out4 = tf.tile(c_out4, [n_s, 1, 1, 1])
		
		# compute mean and stddef of style features first
		s_mean1 = tf.reduce_mean(s_out1, axis=[1,2], keep_dims=True)
		s_mean2 = tf.reduce_mean(s_out2, axis=[1,2], keep_dims=True)
		s_mean3 = tf.reduce_mean(s_out3, axis=[1,2], keep_dims=True)
		s_mean4 = tf.reduce_mean(s_out4, axis=[1,2], keep_dims=True)
		
		s_stddev1 = tf.sqrt(tf.reduce_mean(tf.square(s_out1 - s_mean1), axis=[1,2], keep_dims=True) + 1e-6)
		s_stddev2 = tf.sqrt(tf.reduce_mean(tf.square(s_out2 - s_mean2), axis=[1,2], keep_dims=True) + 1e-6)
		s_stddev3 = tf.sqrt(tf.reduce_mean(tf.square(s_out3 - s_mean3), axis=[1,2], keep_dims=True) + 1e-6)
		s_stddev4 = tf.sqrt(tf.reduce_mean(tf.square(s_out4 - s_mean4), axis=[1,2], keep_dims=True) + 1e-6)
		
		# duplicate style means [s1,s2,s3]=>[s1,s1,...,s2,s2,...,s3,s3,...]
		s_mean1 = tf.expand_dims(s_mean1, axis=1)
		s_mean1 = tf.tile(s_mean1, [1, n_c, 1, 1, 1])
		d = s_mean1.get_shape()[-1].value
		s_mean1 = tf.reshape(s_mean1, [-1, 1, 1, d])
		
		s_mean2 = tf.expand_dims(s_mean2, axis=1)
		s_mean2 = tf.tile(s_mean2, [1, n_c, 1, 1, 1])
		d = s_mean2.get_shape()[-1].value
		s_mean2 = tf.reshape(s_mean2, [-1, 1, 1, d])
		
		s_mean3 = tf.expand_dims(s_mean3, axis=1)
		s_mean3 = tf.tile(s_mean3, [1, n_c, 1, 1, 1])
		d = s_mean3.get_shape()[-1].value
		s_mean3 = tf.reshape(s_mean3, [-1, 1, 1, d])
		
		s_mean4 = tf.expand_dims(s_mean4, axis=1)
		s_mean4 = tf.tile(s_mean4, [1, n_c, 1, 1, 1])
		d = s_mean4.get_shape()[-1].value
		s_mean4 = tf.reshape(s_mean4, [-1, 1, 1, d])
		
		# duplicate style stddev [s1,s2,s3]=>[s1,s1,...,s2,s2,...,s3,s3,...]
		s_stddev1 = tf.expand_dims(s_stddev1, axis=1)
		s_stddev1 = tf.tile(s_stddev1, [1, n_c, 1, 1, 1])
		d = s_stddev1.get_shape()[-1].value
		s_stddev1 = tf.reshape(s_stddev1, [-1, 1, 1, d])
		
		s_stddev2 = tf.expand_dims(s_stddev2, axis=1)
		s_stddev2 = tf.tile(s_stddev2, [1, n_c, 1, 1, 1])
		d = s_stddev2.get_shape()[-1].value
		s_stddev2 = tf.reshape(s_stddev2, [-1, 1, 1, d])
		
		s_stddev3 = tf.expand_dims(s_stddev3, axis=1)
		s_stddev3 = tf.tile(s_stddev3, [1, n_c, 1, 1, 1])
		d = s_stddev3.get_shape()[-1].value
		s_stddev3 = tf.reshape(s_stddev3, [-1, 1, 1, d])
		
		s_stddev4 = tf.expand_dims(s_stddev4, axis=1)
		s_stddev4 = tf.tile(s_stddev4, [1, n_c, 1, 1, 1])
		d = s_stddev4.get_shape()[-1].value
		s_stddev4 = tf.reshape(s_stddev4, [-1, 1, 1, d])
		
		# transform content
		c_mean4 = tf.reduce_mean(c_out4, axis=[1,2], keep_dims=True)
		c_stddev4 = tf.sqrt(tf.reduce_mean(tf.square(c_out4 - c_mean4), axis=[1,2], keep_dims=True) + 1e-6)
		c_out4 = (c_out4 - c_mean4) / c_stddev4 * s_stddev4 + s_mean4
		output = self.decode(c_out4)
		
		# extract features then compute mean and stddev
		if VGG=='VGG16':
			o_out1, o_out2, o_out3, o_out4 = self.vgg16(output, vgg_file=vgg_file)
		else:
			o_out1, o_out2, o_out3, o_out4 = self.vgg19(output, vgg_file=vgg_file)
		
		o_mean1 = tf.reduce_mean(o_out1, axis=[1,2], keep_dims=True)
		o_mean2 = tf.reduce_mean(o_out2, axis=[1,2], keep_dims=True)
		o_mean3 = tf.reduce_mean(o_out3, axis=[1,2], keep_dims=True)
		o_mean4 = tf.reduce_mean(o_out4, axis=[1,2], keep_dims=True)
		
		o_stddev1 = tf.sqrt(tf.reduce_mean(tf.square(o_out1 - o_mean1), axis=[1,2], keep_dims=True))
		o_stddev2 = tf.sqrt(tf.reduce_mean(tf.square(o_out2 - o_mean2), axis=[1,2], keep_dims=True))
		o_stddev3 = tf.sqrt(tf.reduce_mean(tf.square(o_out3 - o_mean3), axis=[1,2], keep_dims=True))
		o_stddev4 = tf.sqrt(tf.reduce_mean(tf.square(o_out4 - o_mean4), axis=[1,2], keep_dims=True))
		
		# compute content_loss
		content_loss = tf.sqrt(tf.reduce_mean(tf.square(o_out4 - c_out4)))
		
		# compute style loss
		style_loss = tf.sqrt(tf.reduce_mean(tf.square(o_mean1 - s_mean1))) + \
			tf.sqrt(tf.reduce_mean(tf.square(o_mean2 - s_mean2))) + \
			tf.sqrt(tf.reduce_mean(tf.square(o_mean3 - s_mean3))) + \
			tf.sqrt(tf.reduce_mean(tf.square(o_mean4 - s_mean4))) + \
			tf.sqrt(tf.reduce_mean(tf.square(o_stddev1 - s_stddev1))) + \
			tf.sqrt(tf.reduce_mean(tf.square(o_stddev2 - s_stddev2))) + \
			tf.sqrt(tf.reduce_mean(tf.square(o_stddev3 - s_stddev3))) + \
			tf.sqrt(tf.reduce_mean(tf.square(o_stddev4 - s_stddev4)))
		
		loss = content_loss + STYLE_LOSS_WEIGHT * style_loss
		return output, loss
	
	def train_on_batch(self, session, content_files, style_files, content_holder, style_holder, cost_holder, train_op):
		contents = []
		for content_file in content_files:
			content = Image.open(content_file)
			content = np.float32(content.convert('RGB'))
			contents.append(content)
		styles = []
		for style_file in style_files:
			style = Image.open(style_file)
			style = np.float32(style.convert('RGB'))
			styles.append(style)
		contents = np.float32(contents)
		styles = np.float32(styles)
		cost_val, _ = session.run(
			[cost_holder, train_op], 
			feed_dict={
				content_holder: contents,
				style_holder: styles})
		return cost_val
		
	def generate_images(self, session, content_files, style_files, content_holder, style_holder, output_holder):
		n_content = len(content_files)
		n_style = len(style_files)
		image = np.zeros([(n_style+1)*IMG_HEIGHT+n_style*10, (n_content+1)*IMG_WIDTH+n_content*10,3], dtype=np.uint8)
		contents = []
		for content_file in content_files:
			content = Image.open(content_file)
			content = np.float32(content.convert('RGB'))
			contents.append(content)
		styles = []
		for style_file in style_files:
			style = Image.open(style_file)
			style = np.float32(style.convert('RGB'))
			styles.append(style)
		contents = np.float32(contents)
		styles = np.float32(styles)
		images = session.run(
			output_holder, 
			feed_dict={
				content_holder: contents,
				style_holder: styles})
		images = np.uint8(images)
		contents = np.uint8(contents)
		styles = np.uint8(styles)
		
		for i in range(n_content):
			start_x = (i+1)*(IMG_WIDTH+10)
			image[0:IMG_HEIGHT, start_x:start_x+IMG_WIDTH] = contents[i]
			
		for j in range(n_style):
			start_y = (j+1)*(IMG_HEIGHT+10)
			image[start_y:start_y+IMG_HEIGHT, 0:IMG_WIDTH] = styles[j]
			
		for j in range(n_style):
			for i in range(n_content):
				start_x = (i+1)*(IMG_WIDTH+10)
				start_y = (j+1)*(IMG_HEIGHT+10)
				image[start_y:start_y+IMG_HEIGHT, start_x:start_x+IMG_WIDTH] = images[j*n_content+i]
		return image
		
	def train(self, content_folder='./content/', style_folder='./style/', vgg_file='./vgg16_notop.h5', num_content=5, num_style=5, num_step=1000, model_path='./model/model', output_folder='./output/', resume=False):
		C = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		S = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		O, cost = self.synthesize(C, S, vgg_file=vgg_file)
		train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
		saver = tf.train.Saver()
		session = tf.Session()
		
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		dataset = self.make_dataset(content_folder=content_folder, style_folder=style_folder)
		n_content = len(dataset['content'])
		n_style = len(dataset['style'])
		count_to_draw = 20
		count_to_save = 0
		mean_cost_val = 0
		for i in range(num_step):
			contents, styles = self.sample_data(dataset, num_content, num_style)
			cost_val = self.train_on_batch(session, contents, styles, C, S, cost, train_op)
			mean_cost_val+=cost_val
			print('Step {:06d}, Cost {:06f}, Count {:06d}'.format(i, cost_val, count_to_save))
			count_to_save += 1
			if count_to_save >= 100:
				saver.save(session, model_path)
				image = self.generate_images(session, contents, styles, C, S, O)
				image = Image.fromarray(image)
				image.save(output_folder + '{:06d}.jpg'.format(count_to_draw))
				mean_cost_val = mean_cost_val/100
				print('=====> Save image, Mean cost {:06f}'.format(mean_cost_val))
				mean_cost_val=0
				count_to_save = 0
				count_to_draw += 1
		session.close()
			

model = Model()
if VGG=='VGG16':
	model.train(
		content_folder='./content_256/',
		style_folder='./style_256/',
		vgg_file='./vgg16_notop.h5',
		num_content=3,
		num_style=3,
		num_step=3000,
		model_path='./model/model_16',
		output_folder='./output_16/',
		resume=True)
else:
	model.train(
		content_folder='./content_256/',
		style_folder='./style_256/',
		vgg_file='./vgg19_notop.h5',
		num_content=3,
		num_style=3,
		num_step=5000,
		model_path='./model/model_19',
		output_folder='./output_19/',
		resume=True)