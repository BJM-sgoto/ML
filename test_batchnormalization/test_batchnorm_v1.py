import os
import tensorflow as tf
import numpy as np
import cv2

class DownSample(tf.keras.Model):
	def __init__(self, filters, kernel_size=(4,4), apply_batchnorm=True):
		super(DownSample,self).__init__()
		self.apply_batchnorm = apply_batchnorm
		self.conv = tf.keras.layers.Conv2D(
			filters,
			kernel_size,
			strides=(2,2),
			padding='same',
			kernel_initializer=tf.random_normal_initializer(0.00,0.02))
		if self.apply_batchnorm:
			self.batchnorm = tf.keras.layers.BatchNormalization()
		
	def call(self, x, training):
		x = self.conv(x)
		if self.apply_batchnorm:
			x = self.batchnorm(x, training)
		x = tf.nn.leaky_relu(x)
		return x
		
class Disc(tf.keras.Model):
	def __init__(self):
		super(Disc,self).__init__()
		self.down1 = DownSample(32, (3,3), apply_batchnorm=False) # 128 X 128
		self.down2 = DownSample(32, (3,3), apply_batchnorm=True) # 64 X 64
		self.down3 = DownSample(64, (3,3), apply_batchnorm=True) # 32 X 32
		self.down4 = DownSample(64, (3,3), apply_batchnorm=True) # 16 X 16
		self.down5 = DownSample(128, (3,3), apply_batchnorm=True) # 8 X 8
		self.down6 = DownSample(128, (3,3), apply_batchnorm=True) # 4 X 4
		
		self.flatten = tf.keras.layers.Flatten()
		
		self.dense8 = tf.keras.layers.Dense(units=32)
		self.dense9 = tf.keras.layers.Dense(units=1)
		
	def call(self, x, training):
		x = self.down1(x, training=training) # 128 X 128
		x = self.down2(x, training=training) # 64 X 64
		x = self.down3(x, training=training) # 32 X 32
		x = self.down4(x, training=training) # 16 X 16
		x = self.down5(x, training=training) # 8 X 8
		x = self.down6(x, training=training) # 4 X 4
		
		x = self.flatten(x)
		x = self.dense8(x)
		x = self.dense9(x)
		return x
		

class Model:
	def make_dataset(self, train_cat_folder='./train_cat/', train_dog_folder='./train_dog/'):
		cat_images = [train_cat_folder + x for x in os.listdir(train_cat_folder)]
		dog_images = [train_dog_folder + x for x in os.listdir(train_dog_folder)]
		outputs = [[0] for i in range(len(cat_images))] + [[1] for i in range(len(dog_images))]
		inputs = cat_images + dog_images
		return inputs, outputs
		
	def shuffle_dataset(self, inputs, outputs):
		ids = np.arange(len(inputs))
		np.random.shuffle(ids)
		new_inputs = [inputs[i] for i in ids]
		new_outputs = [outputs[i] for i in ids]
		return new_inputs, new_outputs

	def init_model(self, image_holder, training):
		self.discriminator = Disc()
		output_holder = self.discriminator(image_holder, training=training)
		return output_holder
		
	def init_model2(self, image_holder, training):
		x = image_holder
		layer_depths = [32,32,64,64,128,128]
		for i in range(len(layer_depths)):
			layer_depth = layer_depths[i]
			x = tf.layers.conv2d(
				x,
				filters=layer_depth,
				kernel_size=(3,3),
				padding='same',
				strides=(2,2))
			if i!=0:
				x = tf.layers.batch_normalization(x, training=training)
			x = tf.nn.leaky_relu(x)
		x = tf.layers.flatten(x)
		x = tf.layers.dense(
			x,
			units=32)
		x = tf.layers.dense(
			x,
			units=1)
		return x

	def train(self, n_epochs=10, batch_size=40, train_cat_folder='./train_cat/', train_dog_folder='./train_dog/', model_path='./model/model', resume=False):
		tf.reset_default_graph()
		X = tf.placeholder(tf.float32, [None, 256, 256, 3])
		Y = tf.placeholder(tf.float32, [None, 1])
		PY = self.init_model(X, training=True)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=Y, 
			logits=PY))
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater(Y, 0.5), tf.greater(PY, 0.5)), dtype=tf.float32))
		optimizer = tf.train.AdamOptimizer()
		grads_vars = optimizer.compute_gradients(loss, self.discriminator.variables)
		train_ops = optimizer.apply_gradients(grads_vars)
		#train_ops = optimizer.minimize(loss)
		inputs, outputs = self.make_dataset(train_cat_folder, train_dog_folder)
		saver = tf.train.Saver(tf.global_variables())
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		n_data = len(inputs)
		for i in range(n_epochs):
			inputs, outputs = self.shuffle_dataset(inputs, outputs)
			for j in range(0, n_data, batch_size):
				end_j = min(j + batch_size, n_data)
				batch_inputs = []
				batch_outputs = []
				for k in range(j, end_j):
					img = cv2.imread(inputs[k])
					h, w, _ = img.shape
					start_cut_x = np.random.randint(low=0, high=w-256+1)
					start_cut_y = np.random.randint(low=0, high=h-256+1)
					img = img[start_cut_y: start_cut_y+256, start_cut_x: start_cut_x+256]
					batch_inputs.append(img)
					batch_outputs.append(outputs[k])
				batch_inputs = np.float32(batch_inputs)/127.5-1
				batch_outputs = np.float32(batch_outputs)
				loss_val, acc_val, _ = session.run([loss, accuracy, train_ops], feed_dict={X: batch_inputs, Y: batch_outputs})
				print('Epoch', i, 'Progress', j, 'Loss', loss_val, 'Accuracy', acc_val)
			saver.save(session, model_path)
		session.close()

	def test(self, test_cat_folder='./test_cat/', test_dog_folder='./test_dog/', model_path='./model/model'):
		tf.reset_default_graph()
		X = tf.placeholder(tf.float32, [None, 256, 256, 3])
		PY = tf.sigmoid(self.init_model(X, training=True))
		cat_images = [test_cat_folder + x for x in os.listdir(test_cat_folder)]
		dog_images = [test_dog_folder + x for x in os.listdir(test_dog_folder)] 
		inputs = cat_images + dog_images
		ids = np.arange(len(inputs))
		np.random.shuffle(ids)
		inputs = [inputs[i] for i in ids]
		batch_size = 5
		session = tf.Session()
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(session, model_path)
		n_data = len(inputs)
		for i in range(0, n_data, batch_size):
			batch_inputs = []
			end_i = min(n_data, i+batch_size)
			for j in range(i, end_i):
				img = cv2.imread(inputs[j])	
				h, w, _ = img.shape
				cut_start_y = (h-256)//2
				cut_start_x = (w-256)//2
				img = img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
				batch_inputs.append(img)
			batch_inputs = np.float32(batch_inputs)/127.5-1
			py_val = session.run(PY, feed_dict={X: batch_inputs})
			for j in range(i, end_i):
				print(inputs[j], py_val[j-i])
		session.close()

	def train2(self, n_epochs=10, batch_size=40, train_cat_folder='./train_cat/', train_dog_folder='./train_dog/', model_path='./model/model', resume=False):
		tf.reset_default_graph()
		X = tf.placeholder(tf.float32, [None, 256, 256, 3])
		Y = tf.placeholder(tf.float32, [None, 1])
		PY = self.init_model2(X, training=True)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=Y, 
			logits=PY))
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater(Y, 0.5), tf.greater(PY, 0.5)), dtype=tf.float32))
		optimizer = tf.train.AdamOptimizer()
		grads_vars = optimizer.compute_gradients(loss, self.discriminator.trainable_variables)
		train_ops = optimizer.apply_gradients(grads_vars)
		#inputs, outputs = self.make_dataset(train_cat_folder, train_dog_folder)
		
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		n_data = len(inputs)
		for i in range(n_epochs):
			inputs, outputs = self.shuffle_dataset(inputs, outputs)
			for j in range(0, n_data, batch_size):
				end_j = min(j + batch_size, n_data)
				batch_inputs = []
				batch_outputs = []
				for k in range(j, end_j):
					img = cv2.imread(inputs[k])
					h, w, _ = img.shape
					start_cut_x = np.random.randint(low=0, high=w-256+1)
					start_cut_y = np.random.randint(low=0, high=h-256+1)
					img = img[start_cut_y: start_cut_y+256, start_cut_x: start_cut_x+256]
					batch_inputs.append(img)
					batch_outputs.append(outputs[k])
				batch_inputs = np.float32(batch_inputs)/127.5-1
				batch_outputs = np.float32(batch_outputs)
				loss_val, acc_val, _ = session.run([loss, accuracy, train_ops], feed_dict={X: batch_inputs, Y: batch_outputs})
				print('Epoch', i, 'Progress', j, 'Loss', loss_val, 'Accuracy', acc_val)
			saver.save(session, model_path)
		session.close()
	
model = Model()


model.train(
	n_epochs=20, 
	batch_size=40, 
	train_cat_folder='./full_train_cat/', 
	train_dog_folder='./full_train_dog/', 
	model_path='./model/model', 
	resume=True)


'''
model.test(
	test_cat_folder='./test_cat/', 
	test_dog_folder='./test_dog/', 
	model_path='./model/model')
'''