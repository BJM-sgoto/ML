import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import random

tf.disable_v2_behavior()
tf.reset_default_graph()

IMG_WIDTH, IMG_HEIGHT = 96, 128
Z_DIM = 128
NUM_ATTR = 40

np.random.seed(1)

class Model:
	def __init__(self):
		self.reuse_encoder = False
		self.reuse_decoder = False
		self.reuse_attribute_classifier = False
		self.reuse_adversarial_classifer = False
		
		self.kernel_initializer = tf.random_normal_initializer(mean=0.00, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()
		
		self.attributes=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Straight_Hair', 'Wavy_Hair', 'Eyeglasses', 'Mouth_Slightly_Open']
	
	def make_fake_attributes(self, num):
		pattern = np.zeros([4,3], dtype=np.float32)
		pattern[0,0]=1
		pattern[1,1]=1
		pattern[2,2]=1
		ids = np.random.randint(size=num, low=0, high=4)
		attribute_hair_color = pattern[ids]
		
		pattern = np.zeros([3,2], dtype=np.float32)
		pattern[0,0]=1
		pattern[1,1]=1
		ids = np.random.randint(size=num, low=0, high=3)
		attribute_hair_shape = pattern[ids]
		
		pattern = np.zeros([2,1], dtype=np.float32)
		pattern[0,0]=1
		ids = np.random.randint(size=num, low=0, high=2)
		attribute_eye = pattern[ids]
		
		ids = np.random.randint(size=num, low=0, high=2)
		attribute_mouth = pattern[ids]
		
		attributes = np.concatenate([attribute_hair_color, attribute_hair_shape, attribute_eye, attribute_mouth], axis=1)
		return attributes
		
	def make_dataset(self, image_folder='./image/', attribute_file='./attribute.txt'):
		dataset = []
		f = open(attribute_file, 'r')
		attrs = f.readline().strip().split(',')
		attrs.pop(0)
		n_attr = len(attrs)
		attr_id = []
		for attribute in self.attributes:
			attr_id.append(attrs.index(attribute))
		s = f.readline()
		n_attr = len(attr_id)
		while s:
			values = s.strip().split(',')
			image_name = values.pop(0)
			attribute_vector = np.zeros(n_attr, dtype=np.float32)
			for i in range(n_attr):
				if values[attr_id[i]]=='1':
					attribute_vector[i] = 1
			dataset.append([image_folder + image_name, attribute_vector])
			s = f.readline()
		f.close()
		return dataset
	
	def encode(self, X):
		with tf.variable_scope('encoder', reuse=self.reuse_encoder):
			Z = X / 127.5 - 1
			blocks = [[32,32],[64,64],[128,128],[256,256]]
			for i, block in enumerate(blocks):
				for layer_depth in block:
					Z = tf.layers.conv2d(Z, filters=layer_depth, kernel_size=3, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=self.kernel_initializer)
				if i!=len(blocks)-1:
					Z = tf.layers.max_pooling2d(Z, pool_size=2, strides=2)
			self.reuse_encoder = True
			return Z
	
	def decode(self, Z, A):
		with tf.variable_scope('decoder', reuse=self.reuse_decoder):
			_, Z_height, Z_width, _ = Z.get_shape().as_list()
			A = tf.layers.dense(A, units=128*Z_height*Z_width, kernel_initializer=self.kernel_initializer, use_bias=False)
			A = tf.reshape(A, [-1, Z_height, Z_width, 128])
			X = tf.concat([Z, A], axis=3)
			blocks = [[256,128],[128,64],[64,32],[32,3]]
			for i, block in enumerate(blocks):
				for layer_depth in block:
					X = tf.layers.conv2d(X, filters=layer_depth, kernel_size=3, padding='same', activation=tf.nn.leaky_relu,kernel_initializer=self.kernel_initializer)
				if i!=0:
					_, X_height, X_width, _ = X.get_shape().as_list()
					X = tf.image.resize(X, (X_height*2, X_width*2))
			X = tf.clip_by_value(X, 0, 255)
			self.reuse_decoder = True
			return X
	
	def attribute_classify(self, X, training=True):
		with tf.variable_scope('attribute_classifier', reuse=self.reuse_attribute_classifier):
			layer_depths = [32,64,128,256]
			Y = X / 127.5 - 1
			for layer_depth in layer_depths:
				Y = tf.layers.conv2d(Y, filters=layer_depth, kernel_size=5, strides=2, padding='same', activation = tf.nn.leaky_relu, kernel_initializer=self.kernel_initializer)
			Y = tf.layers.flatten(Y)
			Y = tf.layers.dropout(Y, 0.4, training=training)
			Y = tf.layers.dense(Y, units=len(self.attributes), activation=tf.nn.sigmoid)
			self.reuse_attribute_classifier = True
			return Y
		
	def adversarial_classify(self, X, training=True):
		with tf.variable_scope('adversarial_classifier', reuse=self.reuse_adversarial_classifer):
			layer_depths = [32,64,128,256]
			Y = X / 127.5 - 1
			for layer_depth in layer_depths:
				Y = tf.layers.conv2d(Y, filters=layer_depth, kernel_size=5, strides=2, padding='same', activation = tf.nn.leaky_relu, kernel_initializer=self.kernel_initializer)
			Y = tf.layers.flatten(Y)
			Y = tf.layers.dropout(Y, 0.4, training=training)
			Y = tf.layers.dense(Y, units=1, activation=tf.nn.sigmoid)
			Y = tf.squeeze(Y, axis=1)
			self.reuse_attribute_classifier = True
			return Y
	
	def compute_loss(self, real_images, reconstructed_images, fake_attributes, predicted_fake_attributes, predicted_outputs):
		loss_construction = tf.reduce_mean(tf.square(real_images/255 - reconstructed_images/255))
		loss_attribute = tf.reduce_mean(tf.square(fake_attributes - predicted_fake_attributes))
		n = tf.cast(tf.shape(predicted_outputs)[0]/2, dtype=tf.int32)
		target_outputs = tf.concat([tf.ones(n, dtype=tf.float32), tf.zeros(n, dtype=tf.float32)], axis=0)
		loss_adversarial = tf.reduce_mean(tf.square(target_outputs - predicted_outputs))
		loss = loss_construction + loss_attribute + loss_adversarial
		return loss
	
	def train(self, num_steps=1000, batch_size=10, image_folder='./image/', attribute_file='./attribute.txt', model_path='./model/model', resume=False):
		
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		RA = tf.placeholder(tf.float32, shape=[None, len(self.attributes)])
		FA = tf.placeholder(tf.float32, shape=[None, len(self.attributes)])
		Z = self.encode(X)
		RX = self.decode(Z, RA)
		FX = self.decode(Z, FA)
		CX = tf.concat([X, FX], axis=0)
		PA = self.attribute_classify(FX, training=True)
		PY = self.adversarial_classify(CX, training=True)
		
		loss = self.compute_loss(X, RX, FA, PA, PY)
		
		train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
			
		dataset = self.make_dataset(image_folder, attribute_file)
		count_to_save = 0
		for i in range(num_steps):
			batch = random.sample(dataset, batch_size)
			x = []
			ra = []
			for item in batch:
				x.append(cv2.imread(item[0]))
				ra.append(item[1])
			x = np.float32(x)
			ra = np.float32(ra)
			fa = self.make_fake_attributes(batch_size)
			loss_val, _ = session.run([loss, train_op], feed_dict={X: x, RA: ra, FA: fa})
			print('Step {:05d}, Loss {:06f}'.format(i, loss_val))
			count_to_save+=1
			if count_to_save>=100:
				saver.save(session, model_path)
				count_to_save = 0
		session.close()
			

model = Model()
model.train(
	num_steps=5000, 
	batch_size=20,
	image_folder='../celab_data/small_celeba/', 
	attribute_file='../celab_data/attribute.csv',
	model_path='./model/model', 
	resume=True)