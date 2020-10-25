import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import cv2

#base_folder = '/content/gdrive/My Drive/machine_learning_data/polygonize/'
base_folder = './'
NUM_EPOCHS = 1000
NUM_SAMPLES = 100
NUM_POINTS = 30 
IMG_SIZE = 128
ANGLE_RESOLUTION = 12
ANGLE_PART = 360 / NUM_POINTS
RADIUS_RESOLUTION = IMG_SIZE/2
NUM_UNITS = 1024
RESUME = False

tf.executing_eagerly()

class Encoder(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(Encoder, self).__init__(**kwargs)
		# layers
		self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.leaky_relu)
		self.pool1 = tf.keras.layers.MaxPool2D()
		self.batchnorm1 = tf.keras.layers.BatchNormalization()
		self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.leaky_relu)
		self.pool2 = tf.keras.layers.MaxPool2D()
		self.batchnorm2 = tf.keras.layers.BatchNormalization()
		self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.leaky_relu)
		self.pool3 = tf.keras.layers.MaxPool2D()
		self.batchnorm3 = tf.keras.layers.BatchNormalization()
		self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation=tf.nn.leaky_relu)
		self.pool4 = tf.keras.layers.MaxPool2D()
		self.batchnorm4 = tf.keras.layers.BatchNormalization()
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(units=NUM_UNITS)

	def call(self, x, training=True):
		y = self.conv1(x)
		y = self.pool1(y)
		y = self.batchnorm1(y, training=training)
		y = self.conv2(y)
		y = self.pool2(y)
		y = self.batchnorm2(y, training=training)
		y = self.conv3(y)
		y = self.pool3(y)
		y = self.batchnorm3(y, training=training)
		y = self.conv4(y)
		y = self.pool4(y)
		y = self.batchnorm4(y, training=training)
		y = self.flatten(y)
		y = self.dense1(y)
		return y

class Decoder(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		#layers
		self.dense_encode_input = tf.keras.layers.Dense(units=NUM_UNITS)
		self.gru = self.make_gru(units=NUM_UNITS)
		self.dense_angle = tf.keras.layers.Dense(units=ANGLE_RESOLUTION)
		self.dense_radius = tf.keras.layers.Dense(units=RADIUS_RESOLUTION)
		
	def make_gru(self, units):
		if tf.test.is_gpu_available():
			return tf.compat.v1.keras.layers.CuDNNGRU(
				units,
				return_sequences=True,
				return_state=True)
		else:
			return tf.keras.layers.GRU(
				units,
				return_sequences=True,
				return_state=True)

	# init_s : output of encoder with shape 1 X self.units
	def call(self, s):
		angle_holder = []
		angle_prob_holder = []
		radius_holder = []
		radius_prob_holder = []
		x = np.zeros([NUM_SAMPLES, 1, NUM_UNITS], dtype=np.float32) # initialize empty input # shape: NUM_SAMPLES X 1 X NUM_OUTPUTS
		
		x, s = self.gru(x, s)
		angle_prob = self.dense_angle(s) # NUM_SAMPLES X ANGLE_RESOLUTION
		angle_prob = tf.nn.softmax(angle_prob, axis=1)
		angle_prob_holder.append(tf.expand_dims(angle_prob, axis=1))
		angle_sampler = tfd.Sample(tfd.OneHotCategorical(probs=angle_prob), sample_shape=[1]) 
		angle = angle_sampler.sample() # NUM_SAMPLES X 1 X ANGLE_RESOLUTION
		angle_holder.append(angle)
		
		radius_prob = self.dense_radius(s) # NUM_SAMPLES X RADIUS_RESOLUTION
		radius_prob = tf.nn.softmax(radius_prob, axis=1)
		radius_prob_holder.append(tf.expand_dims(radius_prob, axis=1))
		radius_sampler = tfd.Sample(tfd.OneHotCategorical(probs=radius_prob), sample_shape=[1])
		radius = radius_sampler.sample() # NUM_SAMPLES X 1 X RADIUS_RESOLUTION
		radius_holder.append(radius)
		x = tf.concat([angle, radius], axis=2)# NUM_SAMPLES X 1 X (ANGLE_RESOLUTION + RADIUS_RESOLUTION)
		x = tf.cast(x, dtype=tf.float32)
		x = self.dense_encode_input(x)
		
		for i in range(1, NUM_POINTS):
			x, s = self.gru(x, s)
			
			angle_prob = self.dense_angle(s) # NUM_SAMPLES X ANGLE_RESOLUTION
			angle_prob = tf.nn.softmax(angle_prob, axis=1)
			angle_prob_holder.append(tf.expand_dims(angle_prob, axis=1))
			angle_sampler = tfd.Sample(tfd.OneHotCategorical(probs=angle_prob), sample_shape=[1]) 
			angle = angle_sampler.sample() # NUM_SAMPLES X 1 X ANGLE_RESOLUTION
			angle_holder.append(angle)
			
			radius_prob = self.dense_radius(s) # NUM_SAMPLES X RADIUS_RESOLUTION
			radius_prob = tf.nn.softmax(radius_prob, axis=1)
			radius_prob_holder.append(tf.expand_dims(radius_prob, axis=1))
			radius_sampler = tfd.Sample(tfd.OneHotCategorical(probs=radius_prob), sample_shape=[1])
			radius = radius_sampler.sample() # NUM_SAMPLES X 1 X RADIUS_RESOLUTION
			radius_holder.append(radius)
			x = tf.concat([angle, radius], axis=2)# NUM_SAMPLES X 1 X (ANGLE_RESOLUTION + RADIUS_RESOLUTION)
			x = tf.cast(x, dtype=tf.float32)
			x = self.dense_encode_input(x)
			
		angle_holder = tf.concat(angle_holder, axis=1) #probability # shape: NUM_SAMPLES X NUM_POINTS X NUM_OUTPUTS
		angle_prob_holder = tf.concat(angle_prob_holder, axis=1)
		radius_holder = tf.concat(radius_holder, axis=1) #probability # shape: NUM_SAMPLES X NUM_POINTS X NUM_OUTPUTS
		radius_prob_holder = tf.concat(radius_prob_holder, axis=1)
		return angle_holder, angle_prob_holder, radius_holder, radius_prob_holder

class Polygonizer(tf.keras.Model):
	def __init__(self, **kwargs):
		super(tf.keras.Model, self).__init__(**kwargs)
		self.full_background = cv2.imread(base_folder + 'background.jpg')
		self.full_background = self.full_background/4 + 255//4*3
		
		self.encoder = Encoder()
		self.decoder = Decoder()
		
	def make_item(self):
		n_points = np.random.randint(low=3, high=8)
		center = [IMG_SIZE//2, IMG_SIZE//2]
		bg_height, bg_width, _ = self.full_background.shape
		start_cut_x = np.random.randint(low=0, high=bg_width-IMG_SIZE)
		end_cut_x = start_cut_x + IMG_SIZE
		start_cut_y = np.random.randint(low=0, high=bg_height-IMG_SIZE)
		end_cut_y = start_cut_y + IMG_SIZE
		img = self.full_background[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
		'''
		angles = np.random.rand(n_points)*np.pi*2
		angles = np.sort(angles)
		
		points = []
		for angle in angles:
			radius = np.random.randint(low=IMG_SIZE*0.3, high=IMG_SIZE*0.45)
			point = [int(center[0] + np.cos(angle)*radius), int(center[1] + np.sin(angle)*radius)]
			points.append(point)
		color = [np.random.randint(low=0, high=255), np.random.randint(low=0, high=255), np.random.randint(low=0, high=255)]
		points = np.int32([points])
		cv2.fillPoly(img, points,[0,0,255])
		mask = np.zeros([IMG_SIZE, IMG_SIZE], dtype=np.float32)
		cv2.fillPoly(mask, points, 255)
		'''
		
		'''
		############ Simple round ############
		cv2.ellipse(img, (64,64), (50,50), 0, 0, 360, (255,0,0), -1)
		mask = np.zeros([IMG_SIZE, IMG_SIZE], dtype=np.float32)
		cv2.ellipse(mask, (64,64), (50,50), 0, 0, 360, 255, -1)
		'''
		
		############ Simple square ############
		cv2.rectangle(img, (34,34), (94,94), (255,0,0), -1)
		mask = np.zeros([IMG_SIZE, IMG_SIZE], dtype=np.float32)
		cv2.rectangle(mask, (34,34), (94,94), (255,0,0), -1)
		
		return img, mask

	def make_multiple_items(self, n_samples):
		imgs = []
		masks = []
		for i in range(n_samples):
			img, mask = self.make_item()
			imgs.append(img)
			masks.append(mask)
		imgs = np.float32(imgs)
		masks = np.float32(masks)
		return imgs, masks

	def compute_iou(self, angle_pos, radius_pos, mask):
		predicted_mask = np.zeros([IMG_SIZE, IMG_SIZE], dtype=np.float32)
		pnts = []
		for i in range(NUM_POINTS):
			radius = radius_pos[i]
			angle = (ANGLE_PART * i + angle_pos[i])/180*np.pi
			x = RADIUS_RESOLUTION + radius * np.cos(angle)
			y = RADIUS_RESOLUTION + radius * np.sin(angle)
			pnts.append([x, y])
		pnts = np.int32([pnts])
		cv2.fillPoly(predicted_mask, [pnts], 255)
		area1 = np.sum(mask)
		area2 = np.sum(predicted_mask)
		intersect = np.where(mask + predicted_mask > 255, 255, 0)
		intersect_area = np.sum(intersect)
		return intersect_area / (area1 + area2 - intersect_area)
	
	def call(self, x, training=True):
		x = self.encoder(x, training=training)
		angle_holder, angle_prob_holder, radius_holder, radius_prob_holder = self.decoder(x)
		return angle_holder, angle_prob_holder, radius_holder, radius_prob_holder

	def compute_loss(self, angle_prob, angle_choice, radius_prob, radius_choice, iou):
		angle_loss = tf.reduce_sum(tf.math.log(tf.reduce_sum(angle_prob * angle_choice, axis=2)), axis=1)
		radius_loss = tf.reduce_sum(tf.math.log(tf.reduce_sum(radius_prob * radius_choice, axis=2)), axis=1)
		loss = -tf.reduce_mean((angle_loss + radius_loss) * iou)
		return loss

model = Polygonizer()
optimizer = tf.keras.optimizers.Adam(5e-4)
for i in range(NUM_EPOCHS):
	imgs, masks = model.make_multiple_items(NUM_SAMPLES)
	imgs = imgs/255
	with tf.GradientTape() as tape:
		angle, angle_prob, radius, radius_prob = model(imgs, training=True)
		angle_choice = angle.numpy()
		radius_choice = radius.numpy()
		angle_prob_val = angle_prob.numpy()
		radius_prob_val = radius_prob.numpy()
		ious = []
		for j in range(NUM_SAMPLES):
			angle_pos = np.where(angle_choice[j]>0.5)[1]
			radius_pos = np.where(radius_choice[j]>0.5)[1]
			iou = model.compute_iou(angle_pos, radius_pos, masks[j])
			ious.append(iou)
		ious = np.float32(ious)
		loss = model.compute_loss(angle_prob, angle_choice, radius_prob, radius_choice, ious)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		cv2.imwrite('./output/mask{:06d}.jpg'.format(i), masks[-1])
		pmask = np.zeros([IMG_SIZE, IMG_SIZE])
		pnts = []
		for j in range(NUM_POINTS):
			radius = radius_pos[j]
			angle = (ANGLE_PART * j + angle_pos[j])/180*np.pi
			x = RADIUS_RESOLUTION + radius * np.cos(angle)
			y = RADIUS_RESOLUTION + radius * np.sin(angle)
			pnts.append([x, y])
		pnts = np.int32([pnts])
		cv2.fillPoly(pmask, [pnts], 255)
		cv2.imwrite('./poutput/pmask{:06d}.jpg'.format(i), pmask)
		print('Loss', loss.numpy(), 'IOU', np.mean(ious))