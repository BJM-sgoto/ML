# reference: https://sthalles.github.io/deep_segmentation_network/
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import os
import random

tf.disable_v2_behavior()
tf.reset_default_graph()

IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 7 + 1 # including unidentified class

LAYERS = {
	'Sky': np.float32([127,0,0]), # 0
	'Cloud': np.float32([255,255,255]), # 1
	'Water': np.float32([255,0,0]), # 2
	'Mountain': np.float32([0,76,76]), # 3
	'Ground': np.float32([76,76,76]), # 4
	'Grass': np.float32([0,255,0]), # 5
	'Tree': np.float32([0,127,0])} # 6

class Model:
	def __init__(self):
		self.kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
		self.bias_initializer = tf.zeros_initializer()
	
	def make_dataset(self, image_folder, segment_folder):
		dataset = []
		for file in os.listdir(image_folder):
			image = image_folder + file
			segment = segment_folder + file
			dataset.append([image, segment])
		return dataset
		
	def sample_data(self, dataset, num):
		ids = list(range(len(dataset)))
		ids = random.sample(ids, num)
		images = []
		segments = []
		for id in ids:
			image = cv2.imread(dataset[id][0])
			segment = cv2.imread(dataset[id][1])[:,:,0]
			h, w, _ = image.shape
			start_y = np.random.randint(low=0, high=h-IMG_HEIGHT+1)
			start_x = np.random.randint(low=0, high=w-IMG_WIDTH+1)
			image = image[start_y: start_y+IMG_HEIGHT, start_x: start_x+IMG_WIDTH]
			segment = segment[start_y: start_y+IMG_HEIGHT, start_x: start_x+IMG_WIDTH]
			segment = cv2.resize(segment, (int(IMG_WIDTH/4), int(IMG_HEIGHT/4)), interpolation=cv2.INTER_NEAREST)
			images.append(image)
			segments.append(segment)
		images = np.float32(images)
		segments = np.int32(segments)
		return images, segments
		
	def atrous_conv2d(self, X, filters, kernel_size=3, rate=2, use_bias=False, name='atrous_conv2d'):
		with tf.variable_scope(name):
			_, h, w, c = X.get_shape().as_list()
			kernel = tf.get_variable('kernel', shape=[kernel_size, kernel_size, c, filters], dtype=tf.float32,initializer=self.kernel_initializer)
			X = tf.nn.atrous_conv2d(X, filters=kernel, rate=rate, padding='SAME')
			if use_bias:
				bias = tf.get_variable('bias', shape=[kernel_size], dtype=tf.float32, initializer=self.bias_initializer)
				X = tf.add_bias(X, bias)
			return X
	
	def atrous_pooling2d(self, X, rates=[1, 2, 4, 8], name='atrous_pooling2d'):
		Xs = []
		with tf.variable_scope(name):
			for i, rate in enumerate(rates):
				if rate==1:
					Xt = self.atrous_conv2d(X, filters=32, kernel_size=1, rate=1, name='atrous_conv2d_' + str(i))
				else:
					Xt = self.atrous_conv2d(X, filters=32, kernel_size=3, rate=rate, name='atrous_conv2d_' + str(i))
				Xs.append(Xt)
			X = tf.concat(Xs, axis=3)
			return X
	
	def segment(self, X, training=False, name='segment'):
		X = X/127.5-1
		with tf.variable_scope(name):
			extractor_layers = [32,64]
			# field : 18
			for layer in extractor_layers:
				X = tf.layers.conv2d(X, filters=layer, kernel_size=3, padding='same', activation=tf.nn.relu)
				X = tf.layers.conv2d(X, filters=layer, kernel_size=3, padding='same', activation=tf.nn.relu)
				X = tf.layers.max_pooling2d(X, strides=2, pool_size=2)
				X = tf.layers.batch_normalization(X, training=training)
			X = tf.layers.conv2d(X, filters=128, kernel_size=3, padding='same')
			# field : 18 + 4*4 = 34
			X = self.atrous_conv2d(X, filters=128, kernel_size=3, rate=2, name='atrous_conv2d_1')
			# field : 34 + 8*4 = 66
			X = self.atrous_conv2d(X, filters=128, kernel_size=3, rate=4, name='atrous_conv2d_2')
			# field : max : 66 + 16*4 = 130
			X = self.atrous_pooling2d(X, rates=[1,2,4,8], name='atrous_pooling2d')
			
			X = tf.layers.dense(X, units=NUM_CLASSES)
			X = tf.nn.softmax(X, axis=3)
			return X
		
	def compute_cost(self, predicted_segmentation, target_segmentation):
		dictionary = np.eye(NUM_CLASSES, dtype=np.float32)
		target_segmentation = tf.expand_dims(target_segmentation, axis=3)
		target_segmentation = tf.gather_nd(dictionary, target_segmentation)
		cost = tf.reduce_mean(tf.square(target_segmentation - predicted_segmentation))
		return cost
			
	def train(self, image_folder='./image/', segment_folder='./segment/', num_steps=1000, batch_size=10, output_folder='./output/', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3])
		Y = tf.placeholder(tf.int32, [None, int(IMG_HEIGHT/4), int(IMG_WIDTH/4)])
		PY = self.segment(X, training=True, name='segment')
		cost = self.compute_cost(PY, Y)
		train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='segment')
		train_op = tf.group([train_op, update_op])
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		dataset = self.make_dataset(image_folder, segment_folder)
		count_to_draw = 0
		count_to_save = 0
		for i in range(num_steps):
			x, y = self.sample_data(dataset, batch_size)
			cost_val, _ = session.run([cost,train_op], feed_dict={X: x, Y: y})
			print('Step {:06d}, Cost {:06f}'.format(i, cost_val))
			count_to_save += 1
			if count_to_save>=100:
				count_to_save = 0
				saver.save(session, model_path)
				count_to_draw +=1
				py = session.run(PY, feed_dict={X: x})
				
				n = min(5, batch_size)
				image = np.zeros([2*IMG_HEIGHT+10, n*IMG_WIDTH+(n-1)*10, 3])
				for j in range(n):
					start_x = j*(IMG_WIDTH+10)
					image[0:IMG_HEIGHT, start_x:start_x+IMG_WIDTH] = x[j]
					ppy = np.float32(np.argmax(py[j], axis=2)*30)
					ppy = cv2.resize(ppy,(IMG_WIDTH,IMG_HEIGHT))
					ppy = np.expand_dims(ppy, axis=2)
					image[IMG_HEIGHT+10:2*IMG_HEIGHT+10, start_x:start_x+IMG_WIDTH] = ppy
				cv2.imwrite(output_folder + 'test_{:06d}.png'.format(count_to_draw), image)
			
		saver.save(session, model_path)
		session.close()
		
model = Model()
model.train(
	image_folder='./image/', 
	segment_folder='./segment/', 
	num_steps=10000, 
	batch_size=10, 
	output_folder='./output/', 
	model_path='./model/model', 
	resume=False)