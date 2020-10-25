import tensorflow.compat.v1 as tf
import numpy as np
import cv2

#base_folder = '/content/gdrive/My Drive/machine_learning_data/polygonize/'
base_folder = './'
NUM_EPOCHS = 1
NUM_SAMPLES = 1
NUM_POINTS = 30 
IMG_SIZE = 128
ANGLE_RESOLUTION = 12
ANGLE_PART = 360 / NUM_POINTS
RADIUS_RESOLUTION = 90
NUM_UNITS = 1024
RESUME = False

tf.disable_v2_behavior()

class Model:
	def __init__(self):
		self.reuse_encoder = False
		self.reuse_decoder = False
		self.full_background = np.float32(cv2.imread('background.jpg'))
		
	def encode(self, input_holder, training=True):
		with tf.variable_scope('encoder', reuse=self.reuse_encoder):
			output_holder = input_holder
			layer_depths = [32,64,128,128,128]
			for layer_depth in layer_depths:
				output_holder = tf.layers.conv2d(
					output_holder,
					kernel_size=(3,3),
					filters=layer_depth,
					padding='same',
					activation=tf.nn.leaky_relu)
				output_holder=tf.layers.max_pooling2d(
					output_holder,
					pool_size=(2,2),
					strides=(2,2))
				output_holder = tf.layers.batch_normalization(
					output_holder,
					training=training)
			output_holder = tf.layers.flatten(output_holder)
			output_holder = tf.layers.dense(
				output_holder,
				units=NUM_UNITS)
			self.reuse_encoder = True
			return output_holder
	
	def decode(self, feature_holder):
		with tf.variable_scope('decoder', reuse=self.reuse_decoder):
			feature_holder = tf.layers.dense(
				feature_holder,
				units=NUM_UNITS)
				
			radius_holder = tf.layers.dense(
				feature_holder,
				units=RADIUS_RESOLUTION)
			radius_holder = tf.nn.softmax(radius_holder, axis=1)
			
			angle_holder = tf.layers.dense(
				feature_holder,
				units=ANGLE_RESOLUTION)
			angle_holder = tf.nn.softmax(angle_holder, axis=1)
			
			self.reuse_decoder=True
			return radius_holder, angle_holder
			
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

	def compute_iou(self, points, mask):
		predicted_mask = np.zeros([IMG_SIZE, IMG_SIZE], dtype=np.float32)
		cv2.fillPoly(predicted_mask, [points], 255)
		area1 = np.sum(mask)
		area2 = np.sum(predicted_mask)
		intersect = np.where(mask + predicted_mask > 255, 255, 0)
		intersect_area = np.sum(intersect)
		cv2.imwrite('predicted_mask.jpg', predicted_mask)
		return intersect_area / (area1 + area2 - intersect_area)

	def compute_cost(self, radius_choices, radius_probs, angle_choices, angle_probs, ious):
		radius_loss = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.reduce_sum(radius_choices * radius_probs, axis=2)), axis=1) * ious)
		angle_loss = tf.reduce_mean(tf.reduce_sum( tf.math.log(tf.reduce_sum(angle_choices * angle_probs, axis=2)), axis=1) * ious)
		loss = radius_loss + angle_loss

	def train(self, model_path='./model/model', resume=False):
		HX = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3+3])
		F = self.encode(HX, training=True)
		R, A = self.decode(F)
		
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		for i in range(NUM_EPOCHS):
			ori_imgs, masks = self.make_multiple_items(NUM_SAMPLES)
			pre_edges = np.zeros([NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 1], dtype=np.float32)
			cur_edges = np.zeros([NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 1], dtype=np.float32)
			signals = np.zeros([NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 1], dtype=np.float32)
			print('--------------\n', ori_imgs.shape, pre_edges.shape, cur_edges.shape, signals.shape)
			imgs = np.concatenate([ori_imgs, pre_edges, cur_edges, signals], axis=3)
			imgs = imgs/255
			points = np.zeros([NUM_SAMPLES, NUM_POINTS, 2], dtype=np.int32)
			ious = np.zeros([NUM_SAMPLES], dtype=np.float32)
			radius_choices = np.zeros([NUM_SAMPLES, NUM_POINTS, RADIUS_RESOLUTION], dtype=np.float32)
			angle_choices = np.zeros([NUM_SAMPLES, NUM_POINTS, ANGLE_RESOLUTION], dtype=np.float32)
			# compute points
			for j in range(NUM_POINTS):
				VR, VA = session.run([R, A], feed_dict={HX: imgs})
				# after the first time, change the signal
				pre_edges = []
				cur_edges = []
				signals = np.zeros([NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 1], dtype=np.float32)*255
				# VR shape: NUM_SAMPLES X RADIUS_RESOLUTION
				# VA shape: NUM_SAMPLES X ANGLE_RESOLUTION
				for k in range(NUM_SAMPLES):
					radius_pos = np.random.choice(RADIUS_RESOLUTION, p=VR[k])
					angle_pos = np.random.choice(ANGLE_RESOLUTION, p=VA[k])
					radius_choices[k,j,radius_pos] = 1
					angle_choices[k,j,angle_pos] = 1
					radius = radius_pos
					angle = (ANGLE_PART * j + angle_pos)/180*np.pi
					points[k, j, 0] = int(IMG_SIZE/2 + radius*np.cos(angle))
					points[k, j, 1] = int(IMG_SIZE/2 + radius*np.sin(angle))
					pre_edge = np.zeros([IMG_SIZE, IMG_SIZE, 1],dtype=np.float32)
					cur_edge = np.zeros([IMG_SIZE, IMG_SIZE, 1],dtype=np.float32)
					if j>=1:
						cur_edge = cv2.polylines(cur_edge, [points[k, j-1:j+1]], False, [255], 1)
						cur_edges.append(cur_edge)
					if j>=2:
						pre_edge = cv2.polylines(pre_edge, [points[k, 0:j]], False, [255], 1)
						pre_edges.append(pre_edge)
			
			cur_edges = np.float32(cur_edges)
			pre_edges = np.float32(pre_edges)
			print('--------------\n', ori_imgs.shape, pre_edges.shape, cur_edges.shape, signals.shape)
			imgs = np.concatenate([ori_imgs, pre_edges, cur_edges, signals], axis=3)
			imgs = imgs/255
			
			# compute iou
			for j in range(NUM_SAMPLES):
				iou = self.compute_iou(points[j], masks[j])
				ious[j] = iou
				
			# compute cost and train	
				
model = Model()
model.train()