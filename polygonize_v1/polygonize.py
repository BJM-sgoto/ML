import tensorflow as tf
import numpy as np
import cv2

#base_folder = '/content/gdrive/My Drive/machine_learning_data/polygonize/'
base_folder = './'

class Polygonizer:
	def __init__(self):
		self.img_size = 128
		self.full_background = cv2.imread(base_folder + 'background.jpg')
		self.full_background = self.full_background/4 + 255//4*3
		self.n_units = 256
		self.START_POLYGON = [1, 0, 0]
		self.POLYGON = [0, 1, 0]
		self.END_POLYGON = [0, 0, 1]

	def make_item(self):
		n_points = np.random.randint(low=3, high=8)
		center = [self.img_size//2 + np.random.randint(low=-20, high=20), 
		self.img_size//2 + np.random.randint(low=-20, high=20)]
		cut_size = np.random.randint(low=50, high=100)
		bg_height, bg_width, _ = self.full_background.shape
		start_cut_x = np.random.randint(low=0, high=bg_width-cut_size)
		end_cut_x = start_cut_x + cut_size
		start_cut_y = np.random.randint(low=0, high=bg_height-cut_size)
		end_cut_y = start_cut_y + cut_size
		img = self.full_background[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
		img = cv2.resize(img, (self.img_size, self.img_size))
		condition = True
		while condition:
			condition = False
			angles = np.random.rand(n_points)*np.pi*2
			angles = np.sort(angles)
			for i in range(n_points):
				d_angle = angles[(i+1)%n_points] - angles[i]
				if d_angle<0:
					d_angle+=np.pi*2
				if d_angle<np.pi/10:
					condition = True
					break
		points = []
		for angle in angles:
			if angle<np.pi/2:
				radius = min(self.img_size - center[0], self.img_size -	center[1])
				radius = np.random.randint(low=int(radius*0.8), high=radius)
			elif angle<np.pi:
				radius = min(center[0], self.img_size -	center[1])
				radius = np.random.randint(low=int(radius*0.8), high=radius)
			elif angle<1.5*np.pi:
				radius = min(center[0], center[1])
				radius = np.random.randint(low=int(radius*0.8), high=radius)
			else:
				radius = min(self.img_size - center[0], center[1])
				radius = np.random.randint(low=int(radius*0.8), high=radius)
			point = [int(center[0] + np.cos(angle)*radius), int(center[1] + np.sin(angle)*radius)]
			points.append(point)
		color = [np.random.randint(low=0, high=255), np.random.randint(low=0, high=255), np.random.randint(low=0, high=255)]
		for i in range(n_points):
			point1 = points[i]
			point2 = points[(i+1)%n_points]
			cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]) , color, 3)
		point = points[0]
		id_minx = 0
		min_x = self.img_size
		for i in range(n_points):
			if points[i][0]<min_x:
				min_x = points[i][0]
				id_minx = i
		points = points[id_minx:] + points[:id_minx]
		return img, points

	def make_multiple_items(self, num):
		imgs = []
		point_sets = [[[0,0]] for i in range(num)]
		is_point = [[self.START_POLYGON] for i in range(num)]
		max_len = 0
		for i in range(num):
			img, points = self.make_item()
			imgs.append(img)
			max_len  = max(max_len, len(points))
			point_sets[i] += points + [[0,0]]
			is_point[i] += [self.POLYGON for i in range(len(points))] + [self.END_POLYGON]
		max_len += 2
		for i in range(num):
			is_point[i] += [self.END_POLYGON for i in range((max_len - len(point_sets[i])))]
			point_sets[i] += [[-1,-1] for i in range((max_len - len(point_sets[i])))]
		imgs = np.float32(imgs)
		point_sets = np.float32(point_sets)
		is_point = np.float32(is_point)
		return imgs, point_sets, is_point
		
	def gru(self, units):
		if tf.test.is_gpu_available():
			return tf.keras.layers.CuDNNGRU(
				units,
				return_sequences=True,
				return_state=True)
		else:
			return tf.keras.layers.GRU(
				units,
				return_sequences=True,
				return_state=True)

	def encode(self, image_holder, training=False):
		# state_holder1 : batch_size X (n_points + 1) X 2(coordinates: x y)
		# state_holder2 : batch_size X (n_points + 1) X 1(is_point)
		state_holder = image_holder
		
		state_size = state_holder.shape
			
		id_x = np.arange(state_size[1].value)/state_size[1].value
		id_x = np.expand_dims(id_x, 1)
		id_x = np.expand_dims(id_x, 2)
		id_x = np.float32(id_x)
		id_x = tf.ones_like(state_holder) * id_x

		id_y = np.arange(state_size[2].value)/state_size[2].value
		id_y = np.expand_dims(id_y, 0)
		id_y = np.expand_dims(id_y, 2)
		id_y = np.float32(id_y)
		id_y = tf.ones_like(state_holder) * id_y

		state_holder = tf.concat([state_holder, id_x, id_y], axis=-1)
		state_holder = tf.layers.batch_normalization(
				state_holder,
				training=training)
		
		layer_depths = [32,64,64,128,128]
		for layer_depth in layer_depths:
			state_holder = tf.layers.conv2d(
				state_holder,
				kernel_size=(3,3),
				filters=layer_depth,
				strides=(1,1),
				padding='same',
				activation=tf.nn.leaky_relu)
			state_holder = tf.layers.max_pooling2d(
				state_holder,
				pool_size=(2,2),
				strides=(2,2))
				
			state_size = state_holder.shape
			
			id_x = np.arange(state_size[1].value)/state_size[1].value
			id_x = np.expand_dims(id_x, 1)
			id_x = np.expand_dims(id_x, 2)
			id_x = np.float32(id_x)
			id_x = tf.ones_like(state_holder) * id_x

			id_y = np.arange(state_size[2].value)/state_size[2].value
			id_y = np.expand_dims(id_y, 0)
			id_y = np.expand_dims(id_y, 2)
			id_y = np.float32(id_y)
			id_y = tf.ones_like(state_holder) * id_y

			state_holder = tf.concat([state_holder, id_x, id_y], axis=-1)
				
			state_holder = tf.layers.batch_normalization(
				state_holder,
				training=training)
				
		state_holder = tf.layers.flatten(state_holder) # size: None X 1024
		
		state_holder = tf.layers.dense(
			state_holder,
			self.n_units)
		return state_holder
		
	def decode(self, state_holder, point_holder, is_point_holder):
	
		input_holder = tf.concat([point_holder, is_point_holder], axis=2)
		input_holder = tf.layers.dense(
			input_holder,
			self.n_units)
	
		output_holder, state_holder = self.gru(self.n_units)(
			input_holder,
			state_holder)
			
		output_point_holder = tf.layers.dense(
			output_holder,
			units=2,
			activation=tf.nn.sigmoid)
			
		output_is_point_holder = tf.layers.dense(
			output_holder,
			units=3)
		output_is_point_holder = tf.nn.softmax(output_is_point_holder)
		
		return output_point_holder, output_is_point_holder, state_holder
		
	def compute_cost(self, point_holder, predicted_point_holder, is_point_holder, predicted_is_point_holder):
		# only one polygon => point_holder.shape = 1 X (n_points+1) X 2
		point_holder_shape = tf.shape(point_holder)
		n_points = point_holder_shape[1] - 1
		mask = tf.concat([tf.ones([point_holder_shape[0], n_points, 2]), tf.zeros([point_holder_shape[0], 1, 2])], axis=1)
		cost1 = tf.reduce_mean(tf.square(point_holder - predicted_point_holder) * mask)
		cost2 = tf.reduce_mean(tf.square(is_point_holder - predicted_is_point_holder))
		cost = cost1 * 10 + cost2*0.1
		cost = tf.Print(cost, [cost1, cost2], 'Cost')
		return cost
		
	def train(self, n_epochs=10, n_samples=1, model_path=base_folder + 'model/model', resume=False):
		tf.reset_default_graph()
		X = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])
		F = self.encode(X, training=True)
		S1 = tf.placeholder(tf.float32, [None, None, 2]) # coordinates
		S2 = tf.placeholder(tf.float32, [None, None, 3]) # is_point
		PY1, PY2, _ = self.decode(F, S1, S2)
		Y1 = tf.placeholder(tf.float32, [None, None, 2])
		Y2 = tf.placeholder(tf.float32, [None, None, 3])
		
		cost = self.compute_cost(Y1, PY1, Y2, PY2)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.train.AdamOptimizer().minimize(cost)
		train_op = tf.group([train_op, update_ops])
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		for i in range(n_epochs):
			imgs, point_sets, is_point = self.make_multiple_items(n_samples)
			point_sets = point_sets/256
			py1, loss, _ = session.run([PY1, cost, train_op], feed_dict={X: imgs, S1: point_sets[:, :-1, :], Y1: point_sets[:, 1:, :], S2: is_point[:,:-1,:], Y2: is_point[:, 1:, :]})
			print('Progress', i, 'Loss', loss)
		saver.save(session, model_path)
		session.close()
		
	def test(self, model_path=base_folder + 'model/model'):
		tf.reset_default_graph()
		X = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])
		F = self.encode(X, training=True)
		print(F)
		FH = tf.placeholder(tf.float32, [None, self.n_units])
		S1 = tf.placeholder(tf.float32, [None, None, 2]) # coordinates
		S2 = tf.placeholder(tf.float32, [None, None, 3]) # is_point
		
		s1 = np.float32([[[0,0]]])
		s2 = np.float32([[[1, 0,0]]])
		imgs, points, _ = self.make_multiple_items(1)
		print(points)
		PY1, PY2, SH = self.decode(FH, S1, S2)
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)
		f = session.run(F, feed_dict={X: imgs})
		img = imgs[0]
		s1, s2,f = session.run([PY1, PY2, SH], feed_dict={FH:f, S1: s1, S2: s2})
		count = 0		
		while np.argmax(s2[0][0])!=2 and count<10:
			print(s1*256, s2)
			new_s1, new_s2, f = session.run([PY1, PY2, SH], feed_dict={FH:f, S1: s1, S2: s2})
			old_p = s1[0][0]
			new_p = new_s1[0][0]
			cv2.line(img, (int(old_p[0]*256), int(old_p[1]*256)), (int(new_p[0]*256), int(new_p[1]*256)), (0,0,255), 3)
			s1 = new_s1
			s2 = new_s2
			count+=1
		session.close()
		cv2.imwrite(base_folder + 'test.jpg', img)
		
model = Polygonizer()
model.train(n_epochs=10000, n_samples=1, resume=True)
#img, point = model.make_item()
#cv2.imwrite(base_folder + 'test.jpg', img)
#model.test()
