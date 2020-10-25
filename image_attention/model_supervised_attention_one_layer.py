import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import random

IMG_WIDTH = 224
IMG_HEIGHT = 224

BATCH_SIZE = 10

RANDOM_SEED = None

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Model:
	def  __init__(self):
		self.reuse_encoder = False
		self.reuse_estimator = False

	def make_dataset(self, image_folder='./image/', mask_folder='./mask/'):
		dataset = []
		for image in os.listdir(image_folder):
			dataset.append([image_folder + image, mask_folder + image])
		return dataset	
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def encode(self, image_input, training=False):
		with tf.variable_scope('encoder', reuse=self.reuse_encoder):
			feature = image_input
			if training:
				noise = tf.random.uniform(tf.shape(feature), minval=-5.0, maxval=5.0, dtype=tf.float32)
				feature = feature + noise
			feature = feature / 255
			local_features = []
			
			layers = [
				{'depth': 32, 'normalization': False, 'pool': False, 'export': False},
				{'depth': 32, 'normalization': True, 'pool': True, 'export': False},
				{'depth': 64, 'normalization': False, 'pool': False, 'export': False},
				{'depth': 64, 'normalization': True, 'pool': True, 'export': False},
				{'depth': 128, 'normalization': False, 'pool': False, 'export': False},
				{'depth': 128, 'normalization': True, 'pool': True, 'export': False},
				{'depth': 128, 'normalization': False, 'pool': False, 'export': False},
				{'depth': 128, 'normalization': True, 'pool': True, 'export': True}, # ==> local feature
				{'depth': 128, 'normalization': False, 'pool': False, 'export': False},
				{'depth': 128, 'normalization': True, 'pool': True, 'export': False}]
			# local feature
			for layer in layers:
				feature = tf.layers.conv2d(
					feature,
					filters=layer['depth'],
					kernel_size=(3,3),
					strides=(1,1),
					padding='same')
				if layer['pool']:
					feature = tf.layers.max_pooling2d(
						feature,
						strides=(2,2),
						pool_size=(2,2))
				if layer['normalization']:
					feature = tf.layers.batch_normalization(feature, training=training)
				if layer['export']:
					local_features.append(feature)
			
			# global feature
			feature = tf.layers.flatten(feature)
			global_feature = tf.layers.dense(
				feature,
				units=512)
			global_feature = tf.layers.batch_normalization(global_feature, training=training)
			
			self.reuse_encoder = True
			return local_features, global_feature
			
	def estimate(self, local_features, global_feature):
		with tf.variable_scope('estimator', reuse=self.reuse_estimator):
			compatibility_scores = []
			
			for local_feature in local_features:
				local_feature_shape = local_feature.get_shape()
				local_feature_height = local_feature_shape[1]
				local_feature_width = local_feature_shape[2]
				local_feature_depth = local_feature_shape[3]
				global_feature_projection = tf.expand_dims(global_feature, axis=1)
				global_feature_projection = tf.expand_dims(global_feature_projection, axis=2)
				global_feature_projection = tf.tile(global_feature_projection, [1, local_feature_height, local_feature_width, 1])
				compatibility_score = tf.concat([local_feature, global_feature_projection], axis=3)
				compatibility_score = tf.layers.dense(
					compatibility_score,
					units=32)
				compatibility_score = tf.layers.dense(
					compatibility_score,
					units=1,
					activation=tf.nn.sigmoid)
				compatibility_score = tf.squeeze(compatibility_score, axis=3)
				compatibility_scores.append(compatibility_score)
			self.reuse_estimator = True
			return compatibility_scores
	
	def compute_cost(self, compatibility_scores, target_mask):
		costs = []
		for compatibility_score in compatibility_scores:
			score_shape = compatibility_score.get_shape()
			score_height = score_shape[1]
			score_width = score_shape[2]
			mask_shape = target_mask.get_shape()
			mask_height = mask_shape[1]
			mask_width = mask_shape[2]
			target_mask = tf.expand_dims(target_mask, axis=3)
			filter_height = int(mask_height//score_height)
			filter_width = int(mask_width//score_width)
			filter = np.ones([filter_height, filter_width, 1, 1], np.float32) / filter_height/ filter_width
			shrunk_mask = tf.nn.conv2d(
				target_mask,
				filter, 
				strides=(filter_height, filter_width),
				padding = 'VALID')
			shrunk_mask = tf.squeeze(shrunk_mask, axis=3)
			cost = tf.reduce_sum(tf.square(compatibility_score - shrunk_mask), axis=(1,2))
			cost = tf.reduce_mean(cost)
			costs.append(cost)
		cost = tf.reduce_sum(costs)
		return cost
	
	def train_on_batch(self, batch, session, tf_images, tf_masks, tf_cost, tf_train_op):
		images = []
		masks = []
		for item in batch:
			image = np.float32(cv2.imread(item[0]))
			image = cv2.resize(image, (IMG_HEIGHT + 20, IMG_WIDTH + 20))
			mask = np.float32(cv2.imread(item[1]))
			mask = cv2.resize(mask, (IMG_HEIGHT + 20, IMG_WIDTH + 20))
			cut_x = np.random.randint(low=0, high=20)
			cut_y = np.random.randint(low=0, high=20)
			image = image[cut_y: cut_y+IMG_HEIGHT, cut_x: cut_x+IMG_WIDTH]
			mask = mask[cut_y: cut_y+IMG_HEIGHT, cut_x: cut_x+IMG_WIDTH]
			mask = np.mean(mask, axis=2)
			mask = np.where(np.greater(mask, 200), 1.0, 0.0)
			images.append(image)
			masks.append(mask)
		images = np.float32(images)
		masks = np.float32(masks)
		cost_val, _ = session.run([tf_cost, tf_train_op], feed_dict={tf_images: images, tf_masks: masks})
		return cost_val
		
	def train(self, batch_size=BATCH_SIZE, num_epoch=100, image_folder='./image/', mask_folder='./mask/', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		LFs, GF = self.encode(X, training=True)
		As = self.estimate(LFs, GF)
		M = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH]) # ==> value within range [0,1]
		cost = self.compute_cost(As, M)
		train_op = tf.train.AdamOptimizer().minimize(cost)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		dataset = self.make_dataset(image_folder=image_folder, mask_folder=mask_folder)
		num_data = len(dataset)
		count_to_save = 0
		for i in range(num_epoch):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(j + batch_size, num_data)
				batch = dataset[j: end_j]
				cost_val = self.train_on_batch(batch, session, X, M, cost, train_op)
				print('Epoch {:03d} Progress {:03d} Loss {:06f}'.format(i, j, cost_val))
				count_to_save+=1
				if count_to_save>=100:	
					count_to_save = 0
					saver.save(session, model_path)
					print('----------------------\nSave model\n----------------------')
		saver.save(session, model_path)
		session.close()
		
	def test(self, batch_size=BATCH_SIZE, image_folder='./test/', output_folder='./output/', model_path='./model/model'):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		LFs, GF = self.encode(X, training=False)
		As = self.estimate(LFs, GF)
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		files = os.listdir(image_folder)
		num_data = len(files)
		for i in range(0, num_data, batch_size):
			end_i = min(i + batch_size, num_data)
			images = []
			for j in range(i, end_i):
				image = np.float32(cv2.imread(image_folder + files[j]))
				image = cv2.resize(image, (IMG_WIDTH + 20, IMG_HEIGHT + 20))
				cut_x = np.random.randint(low=0, high=20)
				cut_y = np.random.randint(low=0, high=20)
				image = image[cut_y: cut_y+IMG_HEIGHT, cut_x: cut_x+IMG_WIDTH]
				images.append(image)
			images = np.float32(images)
			scores = session.run(As, feed_dict={X: images})
			for k, score in enumerate(scores):
				score = score * 255
				for j in range(i, end_i):
					cv2.imwrite(output_folder + files[j][:-3] + '_{:02d}.jpg'.format(k), score[j-i])
		session.close()

model = Model()
'''
model.train(
	batch_size=BATCH_SIZE,
	num_epoch=100,
	image_folder='./image/',
	mask_folder='./mask/',
	model_path='./model/model_supervised_attention_one_layer',
	resume=True)
'''
model.test(
	batch_size=BATCH_SIZE, 
	image_folder='./test_image/', 
	output_folder='./output/model_supervised_attention_one_layer/', 
	model_path='./model/model_supervised_attention_one_layer')