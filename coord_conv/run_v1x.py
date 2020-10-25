# tensorflow 2.0 (style 1.x)
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import os

tf.disable_v2_behavior()

class Model:
	def __init__(self, background_folder):
		self.backgrounds = []
		for x in os.listdir(background_folder):
			img = cv2.imread(background_folder + x)
			img = img/4 + 255*3/4
			self.backgrounds.append(img)
		self.image_size = 128
		
	def regress(self, x, training=False):
		y = x
		layer_depths = [32,32,64,64]
		
		for layer_depth in layer_depths:
			
			y = tf.layers.conv2d(
				y,
				filters=layer_depth, 
				kernel_size=(3,3), 
				padding='same')
			y = tf.layers.max_pooling2d(
				y,
				pool_size=(2,2), 
				strides=(2,2), 
				padding='valid')
			'''
			s = y.get_shape()[1].value
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
			'''
			y = tf.layers.batch_normalization(
				y, 
				training=training)
			
		y = tf.layers.conv2d(
			y,
			filters=8,
			kernel_size=(1,1), 
			padding='same')
		y = tf.layers.flatten(y)
		y = tf.keras.layers.Dense(units=2)(y)
		y = tf.nn.sigmoid(y)
		return y
		
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
	
	def train(self, n_rounds=1000, batch_size=32, model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
		Y = tf.placeholder(tf.float32, [None, 2])
		PY = self.regress(X)
		cost = self.compute_loss(PY, Y)
		train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
			
		for i in range(n_rounds):
			inputs, outputs = self.make_samples(batch_size)
			PY_val, cost_val, _ = session.run([PY, cost, train_op], feed_dict={X: inputs, Y: outputs})
			print('Progress', i, 'Loss', cost_val, 'Val', PY_val[0], outputs[0])
		saver.save(session, model_path)
		
model = Model('./background/')
model.train(
	n_rounds=1000, 
	batch_size=32, 
	model_path='./model/model', 
	resume=False)