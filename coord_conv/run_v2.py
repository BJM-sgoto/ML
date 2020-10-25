# tensorflow version 2.0
import tensorflow as tf
import numpy as np
import os
import cv2

class Model:
	def __init__(self, background_folder):
		self.backgrounds = []
		for x in os.listdir(background_folder):
			img = cv2.imread(background_folder + x)
			img = img/4 + 255*3/4
			self.backgrounds.append(img)
		self.image_size = 128	
		self.regressor = self.make_regressor()
		
	def make_regressor(self):
		x = tf.keras.Input(shape=(self.image_size, self.image_size, 3), name='input_layer')
		y = x
		layer_depths = [32,32,64,64]
		
		for layer_depth in layer_depths:
			y = tf.keras.layers.Conv2D(filters=layer_depth, kernel_size=(3,3), padding='same')(y)
			y = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(y)
			
			s = y.get_shape()[1]
			index_i = np.float32(np.zeros((1, s, s, 1)))
			index_j = np.float32(np.zeros((1, s, s, 1)))
			for i in range(s):
				index_i[0,i] = i
				index_j[0,:,i] = i
			index_i = index_i/self.image_size
			index_j = index_j/self.image_size
			index_i = tf.ones_like(y) * index_i
			index_j = tf.ones_like(y) * index_j
			
			y = tf.concat([y, index_i, index_j], axis=3)
			
			y = tf.keras.layers.BatchNormalization()(y)
			
		y = tf.keras.layers.Conv2D(filters=8, kernel_size=(1,1), padding='same')(y)
		y = tf.keras.layers.Flatten()(y)
		y = tf.keras.layers.Dense(units=2)(y)
		y = tf.nn.sigmoid(y)
		model = tf.keras.Model(inputs=x, outputs=y)
		return model
		
	def make_samples(self, n_samples):
		samples = []
		n_backgrounds = len(self.backgrounds)
		outputs = []
		for i in range(n_samples):
			img = np.copy(self.backgrounds[np.random.randint(low=0, high=n_backgrounds)])
			h, w, _ = img.shape
			start_cut_h = np.random.randint(low=0, high=h-self.image_size+1)
			start_cut_w = np.random.randint(low=0, high=w-self.image_size+1)
			img = img[start_cut_h: start_cut_h + self.image_size, start_cut_w: start_cut_w + self.image_size]
			start_y = np.random.randint(low=0, high=self.image_size-20)
			start_x = np.random.randint(low=0, high=self.image_size-20)
			'''
			color = (np.random.randint(low=0, high=255),
			np.random.randint(low=0, high=255),
			np.random.randint(low=0, high=255))
			'''
			color = (0,0,255)
			img = cv2.rectangle(img, (start_x, start_y), (start_x + 20, start_y + 20), color, -1)
			samples.append(img)
			outputs.append([start_x + 10, start_y + 10])
		return np.float32(samples)/255, np.float32(outputs)/self.image_size 

	def compute_loss(self, x, y):
		return tf.reduce_mean(tf.square(x - y))
	
	def train(self, n_rounds=1000, batch_size=32, model_path='./model/model_v2.ckpt', resume=False):
		self.regressor.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
      loss=self.compute_loss)
		if resume:
			self.regressor.load_weights(model_path)
		for i in range(n_rounds):
			inputs, outputs = self.make_samples(batch_size)
			print('Time', i, ':',  self.regressor.train_on_batch(
				x=inputs,
				y=outputs))
		self.regressor.save_weights(model_path)
		
model = Model('./background/')
model.train(
	n_rounds=1000, 
	batch_size=32, 
	model_path='./model/model_v2.ckpt',
	resume=False)