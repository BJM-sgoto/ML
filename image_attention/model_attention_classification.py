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
		self.num_classes = 2
		self.reuse_encoder = False
		self.reuse_estimator = False
		self.reuse_classifier = False

	def make_dataset(self, image_folder='./image/', mask_folder='./mask/'):
		dataset = []
		sub_folders = os.listdir(image_folder)
		for i, sub_folder in enumerate(sub_folders):
			sub_image_folder_path = image_folder + sub_folder + '/'
			sub_mask_folder_path  =  mask_folder + sub_folder + '/'
			for file in os.listdir(sub_image_folder_path):
				image_path = sub_image_folder_path + file
				mask_path = sub_mask_folder_path + file
				output = np.zeros(self.num_classes, np.float32)
				output[i] = 1
				dataset.append([image_path, mask_path, output])
		
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
	
	def classify(self, local_features, compatibility_scores):
		with tf.variable_scope('classifier', reuse=self.reuse_classifier):
			num_feature = len(local_features)
			class_scores = []
			for i in range(num_feature):
				local_feature = local_features[i]
				compatibility_score = compatibility_scores[i]
				compatibility_score = compatibility_score / (tf.reduce_sum(compatibility_score, axis=(1,2), keepdims=True) + 1e-5)
				compatibility_score = tf.expand_dims(compatibility_score, axis=3)
				class_score = compatibility_score * local_feature
				class_score = tf.layers.flatten(class_score)
				class_score = tf.layers.dense(class_score, units=64)
				class_scores.append(class_score)
			class_scores = tf.concat(class_scores, axis=1)
			class_scores = tf.layers.dense(class_scores, units=64)
			class_scores = tf.layers.dense(class_scores, units=self.num_classes)
			class_scores = tf.nn.softmax(class_scores, axis=-1)
			self.reuse_classifier = True
			return class_scores
	
	def compute_cost(self, compatibility_scores, target_mask, class_scores, target_class_output):
		attention_costs = []
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
			attention_cost = tf.reduce_mean(tf.square(compatibility_score - shrunk_mask), axis=(1,2))
			attention_cost = tf.reduce_mean(attention_cost)
			attention_costs.append(attention_cost)
		attention_cost = tf.reduce_mean(attention_costs)
		classification_cost = tf.reduce_mean(tf.square(class_scores - target_class_output))
		return attention_cost + 0.1 * classification_cost
	
	def compute_accuracy(self, predicted_classes, target_classes):
		predicted_labels = tf.math.argmax(predicted_classes, axis=1)
		target_labels = tf.math.argmax(target_classes, axis=1)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, target_labels), dtype=np.float32))
		return accuracy
	
	def train_on_batch(self, batch, session, tf_images, tf_masks, tf_classes, tf_predicted_classes, tf_cost, tf_accuracy, tf_train_op):
		images = []
		masks = []
		classes = []
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
			classes.append(item[2])
		images = np.float32(images)
		masks = np.float32(masks)
		classes = np.float32(classes)
		cost_val, accuracy_val, predicted_classes, _ = session.run([tf_cost, tf_accuracy, tf_predicted_classes, tf_train_op], 
			feed_dict={
				tf_images: images, 
				tf_masks: masks,
				tf_classes: classes})
		#print('Classes', classes, 'Predicted classes', predicted_classes)
		return cost_val, accuracy_val
		
	def train(self, batch_size=BATCH_SIZE, num_epoch=100, image_folder='./image/', mask_folder='./mask/', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		LFs, GF = self.encode(X, training=True)
		As = self.estimate(LFs, GF)
		PC = self.classify(LFs, As)
		M = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH]) # ==> value within range [0,1]
		C = tf.placeholder(tf.float32, [None, self.num_classes])
		cost = self.compute_cost(As, M, PC, C)
		accuracy = self.compute_accuracy(PC, C)
		
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
				cost_val, accuracy_val = self.train_on_batch(batch, session, X, M, C, PC, cost, accuracy, train_op)
				print('Epoch {:03d} Progress {:03d} Loss {:06f} Acc {:06f}'.format(i, j, cost_val, accuracy_val))
				count_to_save+=1
				if count_to_save>=100:	
					count_to_save = 0
					saver.save(session, model_path)
					print('----------------------\nSave model\n----------------------')
		saver.save(session, model_path)
		session.close()
		
	def test(self, batch_size=BATCH_SIZE, image_folder='./test/', model_path='./model/model'):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		LFs, GF = self.encode(X, training=False)
		As = self.estimate(LFs, GF)
		PC = self.classify(LFs, As)
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
			predicted_classes = session.run(PC, feed_dict={X: images})
			for j in range(i, end_i):
				print(files[j], predicted_classes[j-i])
		session.close()

model = Model()
'''
model.train(
	batch_size=BATCH_SIZE,
	num_epoch=200,
	image_folder='./image/',
	mask_folder='./mask/',
	model_path='./model/model_attention_classification',
	resume=False)
	
'''
model.test(
	batch_size=BATCH_SIZE, 
	image_folder='./temp_test/zebra/', 
	model_path='./model/model_attention_classification')
