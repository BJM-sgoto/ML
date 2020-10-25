# model_text_detector
# only detect lines with text
import tensorflow.compat.v1 as tf
import numpy as np
import random
import os
import cv2
import sys

RANDOM_SEED = 1234

# TRAIN CONSTANTS
N_EPOCH = 100
BATCH_SIZE = 20

# MODEL CONSTANTS
IMG_WIDTH = 512
IMG_HEIGHT = 256
ENCODER_DIM = 128
MIN_SIZE = 10
MAX_SIZE = 40
NUM_SIZE = (MAX_SIZE - MIN_SIZE)//2 + 1 # sizes = [10, 12, ..., 40]
STRIDE = 8 # 

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#np.set_printoptions(threshold=sys.maxsize)

class Model:
	def __init__(self):
		self.reuse_encoder = False
		
		self.min_bound_ys = np.float32(np.zeros([IMG_HEIGHT, NUM_SIZE]))
		self.max_bound_ys = np.float32(np.zeros([IMG_HEIGHT, NUM_SIZE]))
		self.bound_heights = np.float32(np.zeros([IMG_HEIGHT, NUM_SIZE]))
		for i in range(IMG_HEIGHT):
			for j in range(NUM_SIZE):
				size = MIN_SIZE + 2 * j
				self.min_bound_ys[i, j] = i - size/2
				self.max_bound_ys[i, j] = i + size/2
				self.bound_heights[i, j] = size
		
	def make_dataset(self, dataset_folder='./dataset/'):
		dataset = []
		files = os.listdir(dataset_folder)
		image_files = [dataset_folder + file for file in files if file.endswith('png')]
		for image_file in image_files:
			text_file = image_file[:-3] + 'txt'
			f = open(text_file, 'r')
			line = f.readline().strip()
			rectangles = []
			while line:
				subs = line.split('\t')
				rectangles.append(np.int32(eval(subs[1])))
				line = f.readline().strip()
			f.close()
			output = self.compute_iou(rectangles)
			dataset.append([image_file, output])
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
		
	def compute_iou(self, rectangles):
		max_iou = np.float32(np.zeros([IMG_HEIGHT, NUM_SIZE]))
		for rectangle in rectangles:
			line_start_y = rectangle[1]
			line_height = rectangle[3]
			line_end_y = line_start_y + line_height
			
			intersect_start_y = np.maximum(line_start_y, self.min_bound_ys)
			intersect_end_y = np.minimum(line_end_y, self.max_bound_ys)
			intersect_heights = intersect_end_y - intersect_start_y
			intersect_heights = np.where(np.less(intersect_heights,0), 0, intersect_heights)
			
			union_heights = self.bound_heights + line_height - intersect_heights
			
			iou = intersect_heights / union_heights
			max_iou = np.maximum(iou, max_iou)
		return max_iou
		
	def process_batch(self, raw_batch):
		# no augmenting
		images = []
		outputs = []
		for item in raw_batch:
			image = np.float32(cv2.imread(item[0]))
			images.append(image)
			outputs.append(item[1])
		images = np.float32(images)
		outputs = np.float32(outputs)
		return images, outputs
		
	def encode(self, images, encoder_dim, training=False):
		with tf.variable_scope('encoder', reuse=self.reuse_encoder):
			if training:
				noise = tf.random.uniform(tf.shape(images), minval=-5.0, maxval=5.0, dtype=tf.float32)
				images = images + noise
			features = images / 255
			'''
			layers = [
			{'depth':32, 'pool': True, 'batch': True},
			{'depth':32, 'pool': True, 'batch': True},
			{'depth':64, 'pool': True, 'batch': True},
			{'depth':128, 'pool': False, 'batch': False},
			{'depth':256, 'pool': False, 'batch': True}]
			'''
			layers = [
			{'depth':32, 'pool': True, 'batch': True},
			{'depth':32, 'pool': True, 'batch': True},
			{'depth':64, 'pool': True, 'batch': True},
			{'depth':64, 'pool': False, 'batch': False},
			{'depth':128, 'pool': False, 'batch': True}]
			for layer in layers:
				features = tf.layers.conv2d(
					features,
					filters=layer['depth'],
					kernel_size=(3,3),
					strides=(1,1),
					padding='same',
					activation=tf.nn.elu)
				if layer['pool']:
					features = tf.layers.max_pooling2d(
						features,
						strides=(2,2),
						pool_size=(2,2))
				if layer['batch']:
					features = tf.layers.batch_normalization(features, training=training)
			
			# change [batch, height, width, channel] => [batch * height, width, channel] ~ [batch, times, data]
			_, feature_dim1, feature_dim2, feature_dim3 = features.get_shape()
			features = tf.reshape(features, [-1, feature_dim2, feature_dim3])
			encoder_cell = tf.nn.rnn_cell.GRUCell(encoder_dim, name='encoder_cell')
			_, features = tf.nn.dynamic_rnn(
				encoder_cell,
				features,
				dtype=tf.float32,
				time_major=False)
			self.reuse_encoder = True
			features = tf.reshape(features, [-1, feature_dim1,encoder_dim])
			features = tf.layers.dense(
				features,
				units = STRIDE * NUM_SIZE)
			features = tf.reshape(features, [-1, feature_dim1, STRIDE, NUM_SIZE])
			features = tf.reshape(features, [-1, feature_dim1 * STRIDE, NUM_SIZE])
			features = tf.nn.sigmoid(features)
			return features
	
	def train(self, num_epochs=100, batch_size=20, train_folder='./train_dataset', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		Y = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, NUM_SIZE])
		PY = self.encode(X, encoder_dim=ENCODER_DIM, training=True)
		
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y - PY), axis=(1,2)))
		optimizer = tf.train.AdamOptimizer(5e-4)
		train_op = optimizer.minimize(loss)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
					
		dataset = self.make_dataset(dataset_folder=train_folder)
		num_data = len(dataset)
		count_to_save = 0
		print('-------------------------------\nI am here')
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j+batch_size)
				raw_batch = dataset[j: end_j]
				inputs, outputs = self.process_batch(raw_batch)
				_, loss_val = session.run([train_op, loss], feed_dict={X: inputs, Y: outputs})
				print('Epoch {:04d}, Progress {:04d}, Loss {:05f}'.format(i, j, loss_val))
				
				count_to_save += 1
				if count_to_save>=50:
					count_to_save = 0
					saver.save(session, model_path)
					print('====> Save model')
		saver.save(session, model_path)
		session.close()

model = Model()
model.train(
	num_epochs=100,
	batch_size=20,
	train_folder='./train_dataset/',
	model_path='./small_model/model',
	resume=False)