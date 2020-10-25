import tensorflow.compat.v1 as tf
import numpy as np

import os
import random

IMG_WIDTH = 128
IMG_HEIGHT = 128
RANDOM_SEED = 1234

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class Model:
	def __init__(self):	
		self.reuse_forward_encoder = False
		self.reuse_forward_decoder = False
		self.reuse_backward_encoder = False
		self.reuse_backward_decoder = False
		self.reuse_forward_discriminator = False
		self.reuse_backward_discriminator = False
	
	def make_dataset(self, source_folder='./source/', target_folder='./target/'):
		source_files = os.listdir(source_folder)
		source_files = [file for file in source_files if file.endswith('jpg')]
		source_files = [source_folder + file for file in source_files]
		target_files = os.listdir(target_folder)
		target_files = [file for file in target_files if file.endswith('jpg')]
		target_files = [target_folder + file for file in target_files]
		return source_files, target_files
	
	def shuffle_dataset(self, source_files, target_files):
		random.shuffle(source_files)
		random.shuffle(target_files)
	
	def encode(self, images, forward=True, training=False):
		scope = 'backward_encoder'
		reuse = self.reuse_backward_encoder
		if forward:
			scope = 'forward_encoder'
			reuse = self.reuse_forward_encoder
			
		with tf.variable_scope(scope, reuse=self.reuse):
			layers = [{'depth': 32, 'pool': True, 'batch': True, 'export': True},
				{'depth': 64, 'pool': True, 'batch': True, 'export': True},
				{'depth': 128, 'pool': False, 'batch': False, 'export': False},
				{'depth': 128, 'pool': True, 'batch': True, 'export': True},
				{'depth': 128, 'pool': False, 'batch': False, 'export': False},
				{'depth': 128, 'pool': True, 'batch': True, 'export': True},
				{'depth': 128, 'pool': False, 'batch': True, 'export': True}]
			features = []
			feature = images
			for layer in layers:
				feature = tf.layers.conv2d(
					feature,
					kernel_size=(3,3),
					filters=layer['depth'],
					strides=(1,1),
					padding='same',
					activation=tf.nn.elu)
				if layer['pool']:
					feature = tf.layers.max_pooling2d(
						feature,
						pool_size=(2,2),
						strides=(2,2),
						padding='same')
				if layer['batch']:
					feature = tf.layers.batch_normalization(feature, training=training)
				
				if layer['export']:
					features.append(feature)
			if forward:
				self.reuse_forward_encoder = True
			else:
				self.reuse_backward_encoder = True
			return features
		
	def decode(self, features, forward=True, training=False):
		scope = 'backward_decoder'
		reuse = self.reuse_backward_decoder
		if forward:
			scope = 'forward_decoder'
			reuse = self.reuse_forward_decoder
		feature = features
		with tf.variable_scope(scope, reuse=reuse):
			
	
	def backward_encode(self, images):
		with tf.variable_scope('forward_encoder', self.reuse_backward_encoder):
	
	def backward_decode(self, features):
		with tf.variable_scope('forward_encoder', self.reuse_backward_decoder):
		