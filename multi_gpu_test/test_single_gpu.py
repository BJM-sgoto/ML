import tensorflow.compat.v1 as tf
import numpy as np

import os
import cv2
import random
import datetime

IMG_WIDTH = 224
IMG_HEIGHT = 224

RANDOM_SEED = 1234

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class CustomModel:
	def __init__(self):
		self.classes = ['dog', 'cat']
		self.reuse_classifier = False
	
	def make_dataset(self, image_folder, class_names):
		sub_folders = os.listdir(image_folder)
		dataset = []
		for i, class_name in enumerate(class_names):
			if class_name in sub_folders:
				folder_path = image_folder + class_name + '/'
				output = np.zeros(len(class_names), dtype=np.float32)
				output[i] = 1
				for file_name in os.listdir(folder_path):
					dataset.append([folder_path + file_name, np.copy(output)])
			else:
				raise Exception('Class {:s} is not in folder {:s}'.format(class_name, image_folder))
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def classify(self, images, training=False):
		with tf.variable_scope('classifier', reuse = self.reuse_classifier):
			features = images
			if training:
				noise = tf.random.uniform(shape=tf.shape(features), minval=-5.0, maxval=5.0, dtype=tf.float32)
				features = features + noise
				features = tf.where(tf.greater(features, 255), tf.ones_like(features) * 255, features)
				features = tf.where(tf.less(features, 0), tf.zeros_like(features), features)
			features = features / 255
			layers = [
			{'depth':32},
			{'depth':32},
			{'depth':64},
			{'depth':64},
			{'depth':128}]
			for layer in layers:
				features = tf.layers.conv2d(
					features,
					filters=layer['depth'],
					kernel_size=(3,3),
					strides=(1,1),
					padding='same',
					activation=tf.nn.elu)
				features = tf.layers.max_pooling2d(
					features,
					pool_size=(2,2),
					strides=(2,2))
				features = tf.layers.batch_normalization(
					features,
					training=training)
			features = tf.layers.flatten(features)
			features = tf.layers.dense(features, units=64)
			features = tf.layers.batch_normalization(features, training=training)
			features = tf.layers.dense(features, units=len(self.classes))
			features = tf.nn.softmax(features)
			return features
			
	def train_on_batch(self, session, batch, tf_images, tf_outputs, tf_cost, tf_accuracy, tf_train_op):
		images = []
		outputs = []
		start_cpu_time = datetime.datetime.now()
		for item in batch:
			image = cv2.imread(item[0])
			cut_y, cut_x = np.random.randint(size=[2], low=0, high=21)
			image = image[cut_y: cut_y+IMG_HEIGHT, cut_x: cut_x+IMG_WIDTH]
			images.append(image)
			outputs.append(item[1])
		images = np.float32(images)
		outputs = np.float32(outputs)
		end_cpu_time = datetime.datetime.now()
		cpu_time = end_cpu_time - start_cpu_time
		start_gpu_time = datetime.datetime.now()
		loss_val, accuracy_val, _ = session.run(
			[tf_cost, tf_accuracy, tf_train_op], 
			feed_dict={
				tf_images: images,
				tf_outputs: outputs})
		end_gpu_time = datetime.datetime.now()
		gpu_time = end_gpu_time - start_gpu_time
		return loss_val, accuracy_val, cpu_time, gpu_time
			
	def train(self, num_epochs=5, batch_size=40, train_folder='./train/', model_path='./single_gpu/model', resume=False, history_path='./single_gpu/history.txt'):
		X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
		PY = self.classify(X, training=True)
		Y = tf.placeholder(tf.float32, shape=[None, len(self.classes)])
		
		cost = tf.reduce_mean(tf.square(Y - PY))
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, axis=1), tf.argmax(PY, axis=1)), dtype=tf.float32))
		train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		dataset = model.make_dataset(image_folder=train_folder, class_names=self.classes)
		num_data = len(dataset)
		count_to_save = 0
		history_file = open(history_path, 'w')
		history_file.write('Num epochs {:03d}, Batch size {:03d}, Num data {:05d}\n'.format(num_epochs, batch_size, num_data))
		history_file.close()
		mean_loss_val = 0
		mean_accuracy_val = 0
		sum_cpu_time = datetime.timedelta()
		sum_gpu_time = datetime.timedelta()
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j+batch_size)
				batch = dataset[j: end_j]
				loss_val, accuracy_val, cpu_time, gpu_time = self.train_on_batch(session, batch, X, Y, cost, accuracy, train_op)
				mean_loss_val = (mean_loss_val*count_to_save+loss_val)/(count_to_save+1)
				mean_accuracy_val = (mean_accuracy_val*count_to_save+accuracy_val)/(count_to_save+1)
				sum_cpu_time += cpu_time
				sum_gpu_time += gpu_time
				count_to_save+=1
				if count_to_save>=100:
					saver.save(session, model_path)
					history_file = open(history_path, 'a')
					history_file.write('Epc {:03d},Prg {:03d},Lss {:05f},Acc {:05f}, MLss {:05f}, MAcc {:05f},CPU {:s},GPU {:s}\n'.format(i, j, loss_val, accuracy_val, mean_loss_val, mean_accuracy_val, str(sum_cpu_time), str(sum_gpu_time)))
					history_file.close()
					count_to_save = 0
					mean_loss_val = 0
					mean_accuracy_val = 0
				print('Epc {:03d},Prg {:03d},Lss {:05f},Acc {:05f}, MLss {:05f}, MAcc {:05f},CPU {:s},GPU {:s}'.format(i, j, loss_val, accuracy_val, mean_loss_val, mean_accuracy_val, str(sum_cpu_time), str(sum_gpu_time)))
				
		saver.save(session, model_path)
		history_file = open(history_path, 'a')
		history_file.write('Epc {:03d},Prg {:03d},Lss {:05f},Acc {:05f}, MLss {:05f}, MAcc {:05f},CPU {:s},GPU {:s}\n'.format(i, j, loss_val, accuracy_val, mean_loss_val, mean_accuracy_val, str(sum_cpu_time), str(sum_gpu_time)))
		history_file.close()
		session.close()		
			
model = CustomModel()

model.train(
	num_epochs=10, 
	batch_size=20, 
	train_folder='./small_train/', 
	model_path='./single_gpu/model', 
	history_path='./single_gpu/history.txt',
	resume=False)