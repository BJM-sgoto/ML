from PIL import Image
import tensorflow as tf
import os
import random
import numpy as np

class X:
	def __init__(self, img_height=256, img_width=256, img_channel=3, num_epoch=20, learning_rate=1e-4):
		self.img_height = img_height
		self.img_width = img_width
		self.img_channel = img_channel
		self.num_epoch = num_epoch
		self.learning_rate = learning_rate
		current_file = os.path.normpath(os.path.join(dirpath, __file__))
		self.project_folder = current_file[:current_file.rfind('\\')]
		
	def create_dataset(self, rel_image_folder='/train_image/', rel_mark_folder='/mark_folder/'):
		image_folder = os.path.normpath(self.project_folder + rel_image_folder)
		image_files = os.listdir(image_folder)
		mark_folder = os.path.normpath(self.project_folder + rel_mark_folder)
		mark_files = os.listdir(mark_folder)
		marks = []
		images = []
		for i in range(len(images)):
			image_file = images[i]
			mark_file = image_file + '.txt'
			if mark_file in marks:
				images.append(image_folder + image_file)
				marks.append(mark_folder + mark_file)
		return {'image': images, 'mark': marks}
		
	def shuffle_dataset(self):
		
	def init_model(self, input_holder):
		output = input_holder
		layer_depths = [32,32,64,64,128,128]
		with tf.variable_scope('model'):
			for i in range(len(layer_depths)):
			output = tf.layers.conv2d(
				output,
				filters=layer_depths[i],
				kernel_size=(3,3),
				padding='same',
				activation=tf.nn.tanh)
			output = tf.layers.max_pooling2d(
				output,
				pool_size=(2,2),
				strides=(2,2))
			output = tf.layers.batch_normalization(output)
	
	def compute_cost(self):
	
	def train(self):
	def test(self):