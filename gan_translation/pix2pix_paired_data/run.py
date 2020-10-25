import os
import cv2
import tensorflow as tf
import numpy as np

class DownSample(tf.keras.Model):
	def __init__(self, filters, kernel_size=(4,4), apply_batchnorm=True):
		super(DownSample,self).__init__()
		initializer = tf.random_normal_initializer(0., 0.02)
		self.apply_batchnorm = apply_batchnorm
		self.conv = tf.keras.layers.Conv2D(
			filters,
			kernel_size,
			strides=(2,2),
			padding='same',
			use_bias=False,
			kernel_initializer=initializer)
		if self.apply_batchnorm:
			self.batchnorm = tf.keras.layers.BatchNormalization()
		
	def call(self, x, activation, training):
		x = self.conv(x)
		if self.apply_batchnorm:
			x = self.batchnorm(x, training)
		if activation!=None:
			x = activation(x)
		return x
		
class UpSample(tf.keras.Model):
	def __init__(self, filters, kernel_size=(4,4), apply_dropout=True):
		super(UpSample,self).__init__()
		initializer = tf.random_normal_initializer(0., 0.02)
		self.apply_dropout = apply_dropout
		self.up_conv = tf.keras.layers.Conv2DTranspose(
			filters,
			kernel_size,
			strides=(2,2),
			padding='same',
			use_bias=False,
			kernel_initializer=initializer)
		self.batchnorm = tf.keras.layers.BatchNormalization()
		if apply_dropout:
			self.dropout = tf.keras.layers.Dropout(0.5)
		
	def call(self, x1, x2, activation, training):
		x = self.up_conv(x1)
		x = self.batchnorm(x, training=training)
		if self.apply_dropout:
			x = self.dropout(x, training=training)
		if activation!=None:
			x = activation(x)
		if x2!=None:
			x = tf.concat([x, x2], axis=-1)
		return x

class Conv(tf.keras.Model):
	def __init__(self, filters, kernel_size=(3,3), apply_batchnorm=True):
		super(Conv, self).__init__()
		self.apply_batchnorm = apply_batchnorm
		self.conv = tf.keras.layers.Conv2D(
			filters,
			kernel_size,
			padding='same',
			kernel_initializer=tf.random_normal_initializer(0.00,0.02))
		if apply_batchnorm:
			self.batchnorm = tf.keras.layers.BatchNormalization()
	
	def call(self, x, activation, training):
		x = self.conv(x)
		if self.apply_batchnorm:
			x = self.batchnorm(x, training=training)
		if activation!=None:
			x = activation(x)
		return x
		
class Gen(tf.keras.Model):
	def __init__(self):
		super(Gen,self).__init__()
		self.down1 = DownSample(64, (4,4),apply_batchnorm=False) # input is constant => no need batchnorm
		self.down2 = DownSample(128, (4,4), apply_batchnorm=True) # 64 X 64
		self.down3 = DownSample(256, (4,4), apply_batchnorm=True) # 32 X 32
		self.down4 = DownSample(512, (4,4), apply_batchnorm=True) # 16 X 16
		self.down5 = DownSample(512, (4,4), apply_batchnorm=True) # 8 X 8
		self.down6 = DownSample(512, (4,4), apply_batchnorm=True) # 4 X 4
		
		# can use conv here
		self.down7 = DownSample(512, (4,4), apply_batchnorm=True) # 2 X 2
		self.down8 = DownSample(512, (4,4), apply_batchnorm=True) # 1 X 1
		
		self.up9 = UpSample(512, (4,4), apply_dropout=True) # 2 X 2
		self.up10 = UpSample(512, (4,4), apply_dropout=True) # 4 X 4
		self.up11 = UpSample(512, (4,4), apply_dropout=True) # 8 X 8
		self.up12 = UpSample(512, (4,4), apply_dropout=False) # 16 X 16
		self.up13 = UpSample(256, (4,4), apply_dropout=False) # 32 X 32
		self.up14 = UpSample(128, (4,4), apply_dropout=False) # 64 X 64
		self.up15 = UpSample(64, (4,4), apply_dropout=False) # 128 X 128
		initializer = tf.random_normal_initializer(0., 0.02)
		self.last = tf.keras.layers.Conv2DTranspose(
			3,
			(4,4),
			strides=(2,2),
			padding='same',
			kernel_initializer=initializer,
			activation=tf.nn.tanh)
			
	def call(self, x, training):
		x1 = self.down1(x, activation=tf.nn.leaky_relu, training=training) # 128 X 128
		x2 = self.down2(x1, activation=tf.nn.leaky_relu, training=training) # 64 X 64
		x3 = self.down3(x2, activation=tf.nn.leaky_relu, training=training) # 32 X 32
		x4 = self.down4(x3, activation=tf.nn.leaky_relu, training=training) # 16 X 16
		x5 = self.down5(x4, activation=tf.nn.leaky_relu, training=training) # 8 X 8
		x6 = self.down6(x5, activation=tf.nn.leaky_relu, training=training) # 4 X 4
		x7 = self.down7(x6, activation=tf.nn.leaky_relu, training=training) # 2 X 2
		x8 = self.down8(x7, activation=tf.nn.leaky_relu, training=training) # 1 X 1
	
		x9 = self.up9(x8, x7, activation = tf.nn.relu, training=training) # 2 X 2
		x10 = self.up10(x9, x6, activation = tf.nn.relu, training=training) # 4 X 4
		x11 = self.up11(x10, x5, activation = tf.nn.relu, training=training) # 8 X 8
		x12 = self.up12(x11, x4, activation = tf.nn.relu, training=training) # 16 X 16
		x13 = self.up13(x12, x3, activation = tf.nn.relu, training=training) # 32 x 32
		x14 = self.up14(x13, x2, activation = tf.nn.relu, training=training) # 64 x 64
		x15 = self.up15(x14, x1, activation = None, training=training) # 128 X 128
		x16 = self.last(x15) # 256 x 256
		return x16
		
class Disc(tf.keras.Model):
	def __init__(self):
		super(Disc,self).__init__()
		self.down1 = DownSample(32, (4,4), apply_batchnorm=False) # 128 X 128
		self.down2 = DownSample(32, (4,4), apply_batchnorm=True) # 64 X 64
		self.down3 = DownSample(64, (4,4), apply_batchnorm=True) # 32 X 32
		self.down4 = DownSample(64, (4,4), apply_batchnorm=True) # 16 X 16
		self.down5 = DownSample(128, (4,4), apply_batchnorm=True) # 8 X 8
		self.down6 = DownSample(128, (3,3), apply_batchnorm=True) # 4 X 4
		self.conv7 = Conv(32, (3,3), apply_batchnorm=False)
		
		self.last = tf.keras.layers.Dense(units=1)
		
	def call(self, x, training):
		x = self.down1(x, activation=tf.nn.leaky_relu, training=training) # 128 X 128
		x = self.down2(x, activation=tf.nn.leaky_relu, training=training) # 64 X 64
		x = self.down3(x, activation=tf.nn.leaky_relu, training=training) # 32 X 32
		x = self.down4(x, activation=tf.nn.leaky_relu, training=training) # 16 X 16
		x = self.down5(x, activation=tf.nn.leaky_relu, training=training) # 8 X 8
		x = self.down6(x, activation=tf.nn.leaky_relu, training=training) # 4 X 4
		x = self.conv7(x, activation=tf.nn.leaky_relu, training=training)
		x = self.last(x)
		return x		
		
class ShallowDisc(tf.keras.Model):
	def __init__(self):
		super(ShallowDisc, self).__init__()
		self.down1 = DownSample(64, (4,4), apply_batchnorm=False) # constant => not necessary to use batch_norm
		self.down2 = DownSample(128, (4,4), apply_batchnorm=True) # 64 X 64
		self.down3 = DownSample(256, (4,4), apply_batchnorm=True) # 32 # 32
		
		self.conv4 = Conv(512, (4,4), apply_batchnorm=True) # 29 X 29
		self.conv5 = Conv(1, (4,4), apply_batchnorm=False) # 26 X 26 # no batch so the output can get any value
		
	def call(self, x, training):
		x1 = self.down1(x, activation=tf.nn.leaky_relu, training=training)
		x2 = self.down2(x1, activation=tf.nn.leaky_relu, training=training)
		x3 = self.down3(x2, activation=tf.nn.leaky_relu, training=training)
		
		x4 = self.conv4(x3, activation=tf.nn.leaky_relu, training=training)
		x5 = self.conv5(x4, activation=None, training=training)
		return x5

class Model:
	def make_dataset(self, train_folder='./train/'):
		inputs = [train_folder + x for x in os.listdir(train_folder) if x.endswith('png')]
		outputs = [x[:-3] + 'jpg' for x in inputs]
		return inputs, outputs
		
	def shuffle_dataset(self, inputs, outputs):
		ids = np.arange(len(inputs))
		np.random.shuffle(ids)
		new_inputs = [inputs[i] for i in ids]
		new_outputs = [outputs[i] for i in ids]
		return new_inputs, new_outputs
		
	def compute_discriminator_loss(self, disc_real_output, disc_generated_output):
		disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=tf.ones_like(disc_real_output),
			logits=disc_real_output))
		disc_generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=tf.zeros_like(disc_generated_output),
			logits=disc_generated_output))
		return disc_real_loss + disc_generated_loss
	
	def compute_generator_loss(self, disc_generated_output, gen_output, target):
		rev_disc_generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=tf.ones_like(disc_generated_output),
			logits=disc_generated_output))
		gen_loss = tf.reduce_mean(tf.abs(gen_output - target))
		return rev_disc_generated_loss + 100*gen_loss
		#return 100*gen_loss
		
	def train_with_discriminator(self, n_epochs=10, batch_size=10, train_folder='./train/', model_path='./model/model', resume=False):
		inputs, outputs = self.make_dataset(train_folder=train_folder)
		
		X = tf.placeholder(tf.float32, [None, 256, 256, 3])
		RY = tf.placeholder(tf.float32, [None, 256, 256, 3])
		
		generator = Gen()
		discriminator = Disc()
		
		FY = generator(X, training=True) # fake Y
		PFY = discriminator(FY, training=True)
		PRY = discriminator(RY, training=True)
		
		discriminator_loss = self.compute_discriminator_loss(PRY, PFY)
		generator_loss = self.compute_generator_loss(PFY, FY, RY)
		
		generator_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
		generator_grads_vars = generator_optimizer.compute_gradients(generator_loss, generator.variables)
		generator_train_op = generator_optimizer.apply_gradients(generator_grads_vars)
		
		discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
		discriminator_grads_vars = discriminator_optimizer.compute_gradients(discriminator_loss, discriminator.variables)
		discriminator_train_op = discriminator_optimizer.apply_gradients(discriminator_grads_vars)		
		
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
				end_j = min(n_data, j + batch_size)
				batch_inputs = []
				batch_outputs = []
				for k in range(j, end_j):
					input_img = cv2.imread(inputs[k])
					h, w, _ = input_img.shape
					cut_start_y = np.random.randint(low=0, high=h-256+1)
					cut_start_x = np.random.randint(low=0, high=w-256+1)
					input_img = input_img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
					output_img = cv2.imread(outputs[k])
					output_img = output_img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
					batch_inputs.append(input_img)
					batch_outputs.append(output_img)
				batch_inputs = np.float32(batch_inputs) / 127.5 - 1
				batch_outputs = np.float32(batch_outputs) / 127.5 - 1
				
				loss_val, _ = session.run([discriminator_loss, discriminator_train_op], feed_dict={X: batch_inputs, RY: batch_outputs})
				print('Epoch', i, 'Progress', j, 'Discriminator Loss', loss_val)
				predicted_outputs, loss_val, _ = session.run([FY, generator_loss, generator_train_op], feed_dict={X: batch_inputs, RY: batch_outputs})
				print('Epoch', i, 'Progress', j, 'Generator Loss', loss_val)
				
			out_img = predicted_outputs[0] * 127.5 + 127.5
			cv2.imwrite('predict.jpg', out_img)
			out_img = batch_outputs[0] * 127.5 + 127.5
			cv2.imwrite('real.jpg', out_img)
			
			saver.save(session, model_path)
		session.close()
		
	def compute_generator_loss_without_discriminator(self, gen_output, target):
		gen_loss = tf.reduce_mean(tf.abs(gen_output - target))
		return 100*gen_loss

	def train_without_discriminator(self, n_epochs=10, batch_size=10, train_folder='./train/', model_path='./model/model', resume=False):
		inputs, outputs = self.make_dataset(train_folder=train_folder)
		
		X = tf.placeholder(tf.float32, [None, 256, 256, 3])
		RY = tf.placeholder(tf.float32, [None, 256, 256, 3])
		
		generator = Gen()
		
		FY = generator(X, training=True) # fake Y
		
		generator_loss = self.compute_generator_loss_without_discriminator(FY, RY)
		
		generator_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
		generator_grads_vars = generator_optimizer.compute_gradients(generator_loss, generator.variables)
		generator_train_op = generator_optimizer.apply_gradients(generator_grads_vars)
		
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
				end_j = min(n_data, j + batch_size)
				batch_inputs = []
				batch_outputs = []
				for k in range(j, end_j):
					input_img = cv2.imread(inputs[k])
					h, w, _ = input_img.shape
					cut_start_y = np.random.randint(low=0, high=h-256+1)
					cut_start_x = np.random.randint(low=0, high=w-256+1)
					input_img = input_img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
					output_img = cv2.imread(outputs[k])
					output_img = output_img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
					batch_inputs.append(input_img)
					batch_outputs.append(output_img)
				batch_inputs = np.float32(batch_inputs) / 127.5 - 1
				batch_outputs = np.float32(batch_outputs) / 127.5 - 1
				
				predicted_outputs, loss_val, _ = session.run([FY, generator_loss, generator_train_op], feed_dict={X: batch_inputs, RY: batch_outputs})
				print('Epoch', i, 'Progress', j, 'Generator Loss', loss_val)
				
			out_img = predicted_outputs[0] * 127.5 + 127.5
			cv2.imwrite('predict.jpg', out_img)
			out_img = batch_outputs[0] * 127.5 + 127.5
			cv2.imwrite('real.jpg', out_img)
			
			saver.save(session, model_path)
		session.close()
		
	def train_with_shallow_discriminator(self, n_epochs=10, batch_size=10, train_folder='./train/', model_path='./model/model', resume=False):
		inputs, outputs = self.make_dataset(train_folder=train_folder)
		
		X = tf.placeholder(tf.float32, [None, 256, 256, 3])
		RY = tf.placeholder(tf.float32, [None, 256, 256, 3])
		
		generator = Gen()
		discriminator = ShallowDisc()
		
		FY = generator(X, training=True) # fake Y
		PFY = discriminator(FY, training=True)
		PRY = discriminator(RY, training=True)
		
		discriminator_loss = self.compute_discriminator_loss(PRY, PFY)
		generator_loss = self.compute_generator_loss(PFY, FY, RY)
		
		generator_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
		generator_grads_vars = generator_optimizer.compute_gradients(generator_loss, generator.variables)
		generator_train_op = generator_optimizer.apply_gradients(generator_grads_vars)
		
		discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
		discriminator_grads_vars = discriminator_optimizer.compute_gradients(discriminator_loss, discriminator.variables)
		discriminator_train_op = discriminator_optimizer.apply_gradients(discriminator_grads_vars)		
		
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
				end_j = min(n_data, j + batch_size)
				batch_inputs = []
				batch_outputs = []
				for k in range(j, end_j):
					input_img = cv2.imread(inputs[k])
					h, w, _ = input_img.shape
					cut_start_y = np.random.randint(low=0, high=h-256+1)
					cut_start_x = np.random.randint(low=0, high=w-256+1)
					input_img = input_img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
					output_img = cv2.imread(outputs[k])
					output_img = output_img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
					batch_inputs.append(input_img)
					batch_outputs.append(output_img)
				batch_inputs = np.float32(batch_inputs) / 127.5 - 1
				batch_outputs = np.float32(batch_outputs) / 127.5 - 1
				
				loss_val, _ = session.run([discriminator_loss, discriminator_train_op], feed_dict={X: batch_inputs, RY: batch_outputs})
				print('Epoch', i, 'Progress', j, 'Discriminator Loss', loss_val)
				predicted_outputs, loss_val, _ = session.run([FY, generator_loss, generator_train_op], feed_dict={X: batch_inputs, RY: batch_outputs})
				print('Epoch', i, 'Progress', j, 'Generator Loss', loss_val)
				
			out_img = predicted_outputs[0] * 127.5 + 127.5
			cv2.imwrite('predict.jpg', out_img)
			out_img = batch_outputs[0] * 127.5 + 127.5
			cv2.imwrite('real.jpg', out_img)
			
			saver.save(session, model_path)
		session.close()
	
	def test(self, test_folder='./test/', output_folder='./output/', model_path='./model/model', batch_size=10):
		X = tf.placeholder(tf.float32, [None, 256, 256, 3])
		generator = Gen()
		# TODO fix batch normalization
		FY = generator(X, training=True) # fake Y
		saver = tf.train.Saver(tf.global_variables())
		session = tf.Session()
		saver.restore(session, model_path)
		inputs = [test_folder + x for x in os.listdir(test_folder) if x.endswith('png')]
		n_data = len(inputs)
		for i in range(0, n_data, batch_size):
			end_i = min(n_data, i+batch_size)
			batch_inputs = []
			for k in range(i, end_i):
				input_img = cv2.imread(inputs[k])
				h, w, _ = input_img.shape
				cut_start_y = int((h-256)/2)
				cut_start_x = int((w-256)/2)
				input_img = input_img[cut_start_y: cut_start_y+256, cut_start_x: cut_start_x+256]
				batch_inputs.append(input_img)
			batch_inputs = np.float32(batch_inputs)/127.5-1
			batch_outputs = session.run(FY, feed_dict={X: batch_inputs})*127.5 + 127.5
			
			for k in range(i, end_i):
				file_name = output_folder + inputs[k][inputs[k].rfind('/')+1:]
				file_name = file_name[:-3] + 'jpg'
				print(file_name, 'min', np.min(batch_outputs[k-i]), 'max', np.max(batch_outputs[k-i]))
				cv2.imwrite(file_name, batch_outputs[k-i])
			
		session.close()
	
model = Model()
'''
model.train_with_discriminator(
	n_epochs=150, 
	batch_size=10, 
	train_folder='./train/', 
	model_path='./model/model_with_discriminator', 
	resume=False)
'''

'''
model.test(
	test_folder='./test/', 
	output_folder='./output_with_discriminator/', 
	model_path='./model/model_with_discriminator', 
	batch_size=10)
'''

'''
model.train_without_discriminator(
	n_epochs=150, 
	batch_size=15, 
	train_folder='./train/', 
	model_path='./model/model_without_discriminator', 
	resume=True)
'''


model.test(
	test_folder='./test/', 
	output_folder='./output_without_discriminator/', 
	model_path='./model/model_without_discriminator', 
	batch_size=10)


'''
model.train_with_discriminator(
	n_epochs=150, 
	batch_size=10, 
	train_folder='./train/', 
	model_path='./model/model_with_shallow_discriminator', 
	resume=False)
'''

'''
model.test(
	test_folder='./test/', 
	output_folder='./output_with_shallow_discriminator/', 
	model_path='./model/model_with_shallow_discriminator', 
	batch_size=10)
'''