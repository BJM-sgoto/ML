import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2

base_folder = './'
#base_folder = '/content/gdrive/My Drive/machine_learning_data/'
tf.disable_v2_behavior()
tf.reset_default_graph()

class Model:
	def __init__(self, background_folder):
		
		# make background images
		self.backgrounds = []
		background_files = os.listdir(background_folder)
		self.backgrounds = [np.float32(cv2.imread(background_folder + x)) for x in background_files]
		
		# grid: used to compute overlap
		grid_width = 640
		grid_height = 480
		cell_width = 142
		cell_height = 142
		jump_x = 16
		jump_y = 16
		start_center_x = 8
		start_center_y = 8
		n_hcells = int((grid_width - start_center_x)/ jump_y) + 1
		n_vcells = int((grid_height - start_center_y)/ jump_x) + 1
		self.grid = np.float32(np.zeros([n_vcells, n_hcells, 4]))
		self.strip_grid = np.int32(np.zeros([n_vcells, n_hcells, 4]))
		for i in range(n_vcells):
			for j in range(n_hcells):
				temp = start_center_x - cell_width/2 + jump_x*j
				self.grid[i,j,0] = temp
				self.strip_grid[i,j,0] = int(max(temp, 0))
				
				temp = start_center_y - cell_height/2 + jump_y*i
				self.grid[i,j,1] = temp
				self.strip_grid[i,j,1] = int(max(temp, 0))
				
				temp = self.grid[i,j,0] + cell_width
				self.grid[i,j,2] = temp
				self.strip_grid[i,j,2] = int(min(grid_width, temp))
				
				temp = self.grid[i,j,1] + cell_height
				self.grid[i,j,3] = temp				
				self.strip_grid[i,j,3] = int(min(grid_height, temp))
	
	# padding: same - used on big images
	# padding: valid - used on cells
	def extract_features(self, input_holder, padding='same', training=True, add_noise=True):
		output_holder = input_holder
		if add_noise:
			output_holder = output_holder + tf.random.normal(shape=tf.shape(output_holder),mean=0.0,stddev=0.05)
		with tf.variable_scope('extractor'):
			layer_depths = [32,64,128,128]
			# j_out = j_in * s
			# r_out = r_in + (k-1)*j_in
			
			# conv layer k = 3, s = 1
			# j_out = j_in
			# r_out = r_in + 2*j_in
			
			# pool layer k = 2, s = 2
			# j_out = j_in * 2
			# r_out = r_in + j_in
			
			# (j, r)
			# (1, 1)
			# (1, 3) -> (2, 4)
			# (2, 8) -> (4, 10)
			# (4, 18) -> (8, 22)
			# (8, 38) -> (16, 46)
			
			# conv k = 3
			# (16, 78)
			# conv k = 5
			# (16, 142)
			for layer_depth in layer_depths:
				output_holder = tf.layers.conv2d(
					output_holder,
					filters=layer_depth,
					kernel_size=(3,3),
					padding=padding)
				output_holder = tf.layers.max_pooling2d(
					output_holder,
					pool_size=(2,2),
					strides=(2,2),
					padding='valid')
				output_holder = tf.layers.batch_normalization(
					output_holder,
					training=training)
			return output_holder
	
	# padding: same - used on big images
	# padding: valid - used on cells
	def predict_overlap(self, feature_holder, padding='same'):
		overlap_holder = feature_holder
		with tf.variable_scope('overlap_predictor'):
			overlap_holder = tf.layers.conv2d(
				overlap_holder,
				filters = 32,
				kernel_size=(5,5),
				padding=padding)
			overlap_holder = tf.layers.conv2d(
				overlap_holder,
				filters = 32,
				kernel_size=(3,3),
				padding=padding)
			overlap_holder = tf.layers.conv2d(
				overlap_holder,
				filters = 1,
				kernel_size=(1,1),
				padding=padding,
				activation=tf.nn.sigmoid)	
			return overlap_holder
			
	# padding: same - used on big images
	# padding: valid - used on cells
	def predict_size(self, feature_holder, padding='same'):
		overlap_holder = feature_holder
		with tf.variable_scope('size_predictor'):
			overlap_holder = tf.layers.conv2d(
				overlap_holder,
				filters = 32,
				kernel_size=(5,5),
				padding=padding)
			overlap_holder = tf.layers.conv2d(
				overlap_holder,
				filters = 32,
				kernel_size=(3,3),
				padding=padding)
			overlap_holder = tf.layers.conv2d(
				overlap_holder,
				filters = 4,
				kernel_size=(1,1),
				padding=padding,
				activation=tf.nn.sigmoid)	
			return overlap_holder
			
	# padding: same - used on big images
	# padding: valid - used on cells
	def predict_angle(self, feature_holder, padding='same', training=True):
		angle_holder = feature_holder
		angle_holder = tf.layers.dropout(
			angle_holder,
			rate=0.4,
			training=training)
		with tf.variable_scope('angle_predictor'):
			angle_holder = tf.layers.conv2d(
				angle_holder,
				filters = 32,
				kernel_size=(5,5),
				padding=padding)
			angle_holder = tf.layers.dropout(
				angle_holder,
				rate=0.4,
				training=training)
			angle_holder = tf.layers.conv2d(
				angle_holder,
				filters = 32,
				kernel_size=(3,3),
				padding=padding)
			angle_holder = tf.layers.dropout(
				angle_holder,
				rate=0.4,
				training=training)
			angle_holder = tf.layers.conv2d(
				angle_holder,
				filters = 30,
				kernel_size=(1,1),
				padding=padding)
			angle_holder = tf.nn.softmax(angle_holder, axis=3)
			return angle_holder
	
	def compute_overlap(self, rectangle):
		center_x = rectangle[0]
		center_y = rectangle[1]
		rectangle_width = rectangle[2]
		rectangle_height = rectangle[3]
		angle = rectangle[4]
		
		img = np.zeros([480, 640])
		half_width = rectangle_width / 2
		half_height = rectangle_height / 2
		sin_angle = np.sin(angle)
		cos_angle = np.cos(angle)
		x1 = center_x - half_width * cos_angle + half_height * sin_angle
		y1 = center_y - half_width * sin_angle - half_height * cos_angle
		x2 = center_x + half_width * cos_angle + half_height * sin_angle
		y2 = center_y + half_width * sin_angle - half_height * cos_angle
		x3 = center_x + half_width * cos_angle - half_height * sin_angle
		y3 = center_y + half_width * sin_angle + half_height * cos_angle
		x4 = center_x - half_width * cos_angle - half_height * sin_angle
		y4 = center_y - half_width * sin_angle + half_height * cos_angle
		
		pnts = np.int32(np.round([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]))
		img = cv2.fillPoly(img, pnts, 255)
		n_vcells = len(self.strip_grid)
		n_hcells = len(self.strip_grid[0])
		ret = np.float32(np.zeros([n_vcells, n_hcells]))
		area = np.sum(img)
		for i in range(n_vcells):
			for j in range(n_hcells):
				ret[i,j] = np.sum(img[self.strip_grid[i,j,1]: self.strip_grid[i,j,3], self.strip_grid[i,j,0]: self.strip_grid[i,j,2]])/area
		#ret = np.where(ret>0.9, 1, 0)
		return ret
	
	def make_dataset(self, image_folder, label_folder):
		label_files = os.listdir(label_folder)
		label_files = [x for x in label_files if x.endswith('.txt')]
		inputs = [image_folder + x[:-4] for x in label_files]
		outputs = []
		for label_file in label_files:
			f = open(label_folder + label_file, 'r')
			s = f.read().strip().split()
			f.close()
			t_output = [float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4])]
			outputs.append(t_output)
		return {'input': inputs, 'output': outputs}	
		
	def shuffle_dataset(self, dataset):
		ids = np.arange(len(dataset['input']))
		np.random.shuffle(ids)
		inputs = [dataset['input'][i] for i in ids]
		outputs = [dataset['output'][i] for i in ids]
		return {'input': inputs, 'output': outputs}
	
	# agument data
	# img, background must be float
	def rotate_translate_image(self, img, rectangle):
		background = self.backgrounds[np.random.randint(low=0, high=len(self.backgrounds))]
		h, w, _ = background.shape
		start_cut_x = np.random.randint(low=0, high=w-640+1)
		end_cut_x = start_cut_x + 640
		start_cut_y = np.random.randint(low=0, high=h-480+1)
		end_cut_y = start_cut_y + 480
		background = background[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
		background = background.copy()
		
		rotate_angle = np.random.randint(low=0, high=360)
		M = cv2.getRotationMatrix2D((rectangle[0], rectangle[1]),rotate_angle,1)
		dx, dy = np.random.randint(low=-7, high=8, size=[2])
		M += np.float32([[0, 0, dx],[0, 0, dy]])
		img = img + 1
		img = cv2.warpAffine(img, M, (640, 480))
		mask = np.where(img>0.5, 1, 0)
		img = (img - 1) * mask + background * (1 - mask)
		angle = rectangle[4] - rotate_angle / 180 * np.pi
		if angle < 0:
			angle += 2*np.pi
		return img, angle, dx, dy
	
	def compute_overlap_loss(self, overlap, predicted_overlap, has_object):
		ones_like_has_object = tf.ones_like(has_object)
		zeros_like_has_object = tf.zeros_like(has_object)
		object_rate = tf.reduce_mean(has_object)
		train_cells = tf.random.uniform(shape=tf.shape(has_object), minval=0, maxval=1)
		train_cells = tf.where(tf.greater(train_cells, 1 - 3*object_rate), ones_like_has_object, zeros_like_has_object)
		train_cells = train_cells + has_object
		train_cells = tf.where(tf.greater(train_cells, 0.5), ones_like_has_object, zeros_like_has_object)
		return tf.reduce_sum(tf.square(overlap - predicted_overlap)*train_cells)/ (tf.reduce_sum(train_cells) + 1e-6)
		#return tf.reduce_mean(tf.square(overlap - predicted_overlap))
		
	def compute_size_loss(self, size, predicted_size, has_object):
		return tf.reduce_sum(tf.square(size - predicted_size)*has_object)/(tf.reduce_sum(has_object)*4 + 1e-6)
	
	def compute_angle_loss(self, angle, predicted_angle, has_object):
		return tf.reduce_sum(tf.square(angle - predicted_angle)*has_object)/(tf.reduce_sum(has_object)*30 + 1e-6)
	
	def train(self,image_folder='./image/', label_folder='./label/', n_epoch=100, batch_size=10, model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, shape=[None, 480, 640, 3])
		Y1 = tf.placeholder(tf.float32, shape=[None, 30, 40, 1])
		Y2 = tf.placeholder(tf.float32, shape=[None, 30, 40, 4])
		Y3 = tf.placeholder(tf.float32, shape=[None, 30, 40, 30])
		F = self.extract_features(X, training=True, padding='same', add_noise=True)
		PY1 = self.predict_overlap(F, padding='same')
		PY2 = self.predict_size(F, padding='same')
		PY3 = self.predict_angle(F, training=True, padding='same')
		has_object = tf.where(tf.greater(Y1, 0.9), tf.ones_like(Y1), tf.zeros_like(Y1))
		overlap_loss = self.compute_overlap_loss(Y1, PY1, has_object)
		size_loss =  self.compute_size_loss(Y2, PY2, has_object)
		angle_loss = self.compute_angle_loss(Y3, PY3, has_object)
		loss = overlap_loss + size_loss + angle_loss
		with tf.variable_scope('main_optimizer'):
			#varlist = tf.trainable_variables('extractor') + tf.trainable_variables('overlap_predictor') + tf.trainable_variables('size_predictor')
			train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)
			update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group([train_op, update_op])
		with tf.variable_scope('overlap_optimizer'):
			overlap_train_op = tf.train.AdamOptimizer(5e-4).minimize(overlap_loss, var_list=tf.trainable_variables('overlap_predictor'))
		with tf.variable_scope('size_optimizer'):
			size_train_op = tf.train.AdamOptimizer(5e-4).minimize(size_loss, var_list=tf.trainable_variables('size_predictor'))	
		with tf.variable_scope('angle_optimizer'):
			angle_train_op = tf.train.AdamOptimizer(5e-4).minimize(angle_loss, var_list=tf.trainable_variables('angle_predictor'))
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		dataset = self.make_dataset(image_folder, label_folder)
		n_data = len(dataset['input'])
		for i in range(n_epoch):
			print('i', i, n_data)
			dataset = self.shuffle_dataset(dataset)
			for j in range(0, n_data,batch_size):
				print('j', j)
				x = []
				y1 = []
				y2 = []
				y3 = []
				end_j = min(n_data, j + batch_size)
				for k in range(j, end_j):
					img = np.float32(cv2.imread(dataset['input'][k]))
					rectangle = np.copy(dataset['output'][k])
					img, angle, dx, dy = self.rotate_translate_image(img, rectangle)
					x.append(img)
					
					rectangle[4] = angle
					rectangle[0] += dx
					rectangle[1] += dy
					overlap = self.compute_overlap(rectangle)
					y1.append(np.expand_dims(overlap, axis=2))
					
					size = np.zeros([30, 40, 4])
					size[:,:,0] = rectangle[0] - self.grid[:,:,0]
					size[:,:,1] = rectangle[1] - self.grid[:,:,1]
					size[:,:,2] = rectangle[2]
					size[:,:,3] = rectangle[3]
					y2.append(size)
					
					angle_output = np.zeros([30,40,30])
					if angle>np.pi:
						angle=angle-np.pi
					angle_output[:, :, int(angle*30/np.pi)] = 1
					y3.append(angle_output)
					
				x = np.float32(x) / 255
				y1 = np.float32(y1)
				y2 = np.float32(y2) / 142
				y3 = np.float32(y3)
				_, overlap_loss_val, size_loss_val, angle_loss_val = session.run([train_op, overlap_loss, size_loss, angle_loss], feed_dict={X: x, Y1: y1, Y2: y2, Y3: y3})
				print('Epoch', i, 'Progress', j, 'Loss', overlap_loss_val, size_loss_val, angle_loss_val)
				
			saver.save(session, model_path)
		session.close()
	
	def train_overlap_predictor(self, image_folder='./image/', label_folder='./label/', n_epoch=100, batch_size=10, model_path='./model/model'):
		X = tf.placeholder(tf.float32, shape=[None, 480, 640, 3])
		Y1 = tf.placeholder(tf.float32, shape=[None, 30, 40, 1])
		Y2 = tf.placeholder(tf.float32, shape=[None, 30, 40, 4])
		Y3 = tf.placeholder(tf.float32, shape=[None, 30, 40, 30])
		F = self.extract_features(X, training=False, add_noise=True)
		PY1 = self.predict_overlap(F)
		PY2 = self.predict_size(F)
		PY3 = self.predict_angle(F)
		
		has_object = tf.where(tf.greater(Y1, 0.9), tf.ones_like(Y1), tf.zeros_like(Y1))
		#overlap_loss =  self.compute_overlap_loss(Y1, PY1, has_object)
		overlap_loss =  tf.reduce_mean(tf.square(Y1 - PY1))
		size_loss =  self.compute_size_loss(Y2, PY2, has_object)
		angle_loss = self.compute_angle_loss(Y3, PY3, has_object)
		loss = overlap_loss + size_loss + angle_loss
		with tf.variable_scope('main_optimizer'):
			#varlist = tf.trainable_variables('extractor') + tf.trainable_variables('overlap_predictor') + tf.trainable_variables('size_predictor')
			train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)
			update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group([train_op, update_op])
		with tf.variable_scope('overlap_optimizer'):
			overlap_train_op = tf.train.AdamOptimizer(5e-4).minimize(overlap_loss, var_list=tf.trainable_variables('overlap_predictor'))
		with tf.variable_scope('size_optimizer'):
			size_train_op = tf.train.AdamOptimizer(5e-4).minimize(size_loss, var_list=tf.trainable_variables('size_predictor'))	
		with tf.variable_scope('angle_optimizer'):
			angle_train_op = tf.train.AdamOptimizer(5e-4).minimize(angle_loss, var_list=tf.trainable_variables('angle_predictor'))
		
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)		
		
		dataset = self.make_dataset(image_folder, label_folder)
		n_data = len(dataset['input'])
		for i in range(n_epoch):
			dataset = self.shuffle_dataset(dataset)
			for j in range(0, n_data,batch_size):
				x = []
				y1 = []
				end_j = min(n_data, j + batch_size)
				for k in range(j, end_j):
					img = np.float32(cv2.imread(dataset['input'][k]))
					rectangle = np.copy(dataset['output'][k])
					img, angle, dx, dy = self.rotate_translate_image(img, rectangle)
					x.append(img)
					
					rectangle[4] = angle
					rectangle[0] += dx
					rectangle[1] += dy
					overlap = self.compute_overlap(rectangle)
					y1.append(np.expand_dims(overlap, axis=2))
					
				x = np.float32(x) / 255
				y1 = np.float32(y1)
				_, overlap_loss_val = session.run([overlap_train_op, overlap_loss], feed_dict={X: x, Y1: y1})
				print('Epoch', i, 'Progress', j, 'Loss', overlap_loss_val)
				
			saver.save(session, model_path)
		session.close()
	
	def train_size_angle_predictor(self, image_folder='./image/', label_folder='./label/', n_epoch=100, batch_size=10, model_path='./model/model'):
		X = tf.placeholder(tf.float32, shape=[None, 142, 142, 3])
		Y1 = tf.placeholder(tf.float32, shape=[None, 1])
		Y2 = tf.placeholder(tf.float32, shape=[None, 4])
		Y3 = tf.placeholder(tf.float32, shape=[None, 30])
		F = self.extract_features(X, training=False, padding='valid', add_noise=True)
		PY1 = self.predict_overlap(F, padding='valid')
		PY2 = self.predict_size(F, padding='valid')
		PY3 = self.predict_angle(F, padding='valid')
		PY1 = tf.squeeze(PY1, axis=[1,2])
		PY2 = tf.squeeze(PY2, axis=[1,2])
		PY3 = tf.squeeze(PY3, axis=[1,2])
		
		overlap_loss =  tf.reduce_mean(tf.square(Y1 - PY1))
		size_loss =  tf.reduce_mean(tf.square(Y2 - PY2))
		angle_loss = tf.reduce_mean(tf.square(Y3 - PY3))
		loss = overlap_loss + size_loss + angle_loss
		with tf.variable_scope('main_optimizer'):
			#varlist = tf.trainable_variables('extractor') + tf.trainable_variables('overlap_predictor') + tf.trainable_variables('size_predictor')
			train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)
			update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group([train_op, update_op])
		with tf.variable_scope('overlap_optimizer'):
			overlap_train_op = tf.train.AdamOptimizer(5e-4).minimize(overlap_loss, var_list=tf.trainable_variables('overlap_predictor'))
		with tf.variable_scope('size_optimizer'):
			size_train_op = tf.train.AdamOptimizer(5e-4).minimize(size_loss, var_list=tf.trainable_variables('size_predictor'))	
		with tf.variable_scope('angle_optimizer'):
			angle_train_op = tf.train.AdamOptimizer(5e-4).minimize(angle_loss, var_list=tf.trainable_variables('angle_predictor'))
		
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)
		
		#session.run(tf.global_variables_initializer())
		dataset = self.make_dataset(image_folder, label_folder)
		n_data = len(dataset['input'])
		for i in range(n_epoch):
			dataset = self.shuffle_dataset(dataset)
			for j in range(0, n_data, batch_size):
				end_j = min(n_data, j+batch_size)
				x = []
				y2 = []
				y3 = []
				for k in range(j, end_j):
					y2.append(dataset['output'][k][:-1])
					
					img = np.float32(cv2.imread(dataset['input'][k]))
					x.append(img)
					angle_output = np.float32(np.zeros([30]))
					angle=dataset['output'][k][4]
					if angle>np.pi:
						angle=angle-np.pi
					angle_output[int(angle*30/np.pi)] = 1
					y3.append(angle_output)
					
				#print(x)
				x = np.float32(x)/255
				y2 = np.float32(y2)/142
				y3 = np.float32(y3)
				_, size_loss_val, _, angle_loss_val = session.run([size_train_op, size_loss, angle_train_op, angle_loss], feed_dict={X: x, Y2: y2, Y3: y3})
				print('Epoch', i, 'Progress', j, 'Loss', size_loss_val, angle_loss_val)
			saver.save(session, model_path)
		session.close()
		
	def test(self, test_folder='./image/', output_folder='./output/', batch_size=10, model_path='./model/model'):
		test_files = os.listdir(test_folder)
		X = tf.placeholder(tf.float32, shape=[None, 480, 640, 3])
		F = self.extract_features(X, training=False, padding='same', add_noise=False)
		PY1 = self.predict_overlap(F, padding='same')
		PY2 = self.predict_size(F, padding='same')
		PY3 = self.predict_angle(F, padding='same', training=False)
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)
		n_data = len(test_files)
		for i in range(0, n_data, batch_size):
			imgs = []
			end_i = min(i+batch_size, n_data)
			for j in range(i, end_i):
				imgs.append(cv2.imread(test_folder + test_files[j]))
			imgs = np.float32(imgs)
			overlap, size, angle = session.run([PY1, PY2, PY3], feed_dict={X: imgs/255})
			for j in range(i, end_i):
				img = imgs[j-i]
				for m in range(30):
					for n in range(40):
						t_size = size[j-i, m, n]*142
						if overlap[j-i, m, n, 0]>0.98 and 55<=t_size[0]<=87 and 55<=t_size[1]<=87:
							t_angle = angle[j-i, m, n]
							t_angle = (np.where(t_angle==np.max(t_angle))[0][0] + 0.5)*np.pi/30
							center_x = t_size[0] + self.grid[m,n,0]
							center_y = t_size[1] + self.grid[m,n,1]
							t_half_width = t_size[2] / 2
							t_half_height = t_size[3] / 2
							pnts = [
								[center_x-t_half_width*np.cos(t_angle)+t_half_height*np.sin(t_angle),center_y-t_half_width*np.sin(t_angle)-t_half_height*np.cos(t_angle)],
								[center_x+t_half_width*np.cos(t_angle)+t_half_height*np.sin(t_angle),center_y+t_half_width*np.sin(t_angle)-t_half_height*np.cos(t_angle)],
								[center_x+t_half_width*np.cos(t_angle)-t_half_height*np.sin(t_angle),center_y+t_half_width*np.sin(t_angle)+t_half_height*np.cos(t_angle)],
								[center_x-t_half_width*np.cos(t_angle)-t_half_height*np.sin(t_angle),center_y-t_half_width*np.sin(t_angle)+t_half_height*np.cos(t_angle)]]
							pnts = np.int32([pnts])
							img = cv2.polylines(img, pnts, True, (255,0,0), 1)
							img = cv2.rectangle(img, (self.grid[m,n,0],self.grid[m,n,1]), (self.grid[m,n,2],self.grid[m,n,3]), (255,0,0), 1)
				cv2.imwrite(output_folder + test_files[j], img)
				print(output_folder + test_files[j])
		session.close()

	def test_overlap_predictor(self, test_folder='./test/', batch_size=10, model_path='./model/model'):
		test_files = os.listdir(test_folder)
		X = tf.placeholder(tf.float32, shape=[None, 480, 640, 3])
		F = self.extract_features(X, training=False, padding='same', add_noise=False)
		PY1 = self.predict_overlap(F, padding='same')
		PY2 = self.predict_size(F, padding='same')
		PY3 = self.predict_angle(F, padding='same', add_noise=True)
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)
		n_data = len(test_files)
		for i in range(0, n_data, batch_size):
			imgs = []
			end_i = min(i+batch_size, n_data)
			for j in range(i, end_i):
				imgs.append(cv2.imread(test_folder + test_files[j]))
			imgs = np.float32(imgs)
			overlap, size, angle = session.run([PY1, PY2, PY3], feed_dict={X: imgs/255})
			for j in range(i, end_i):
				img = imgs[j-i]
				for m in range(30):
					for n in range(40):
						if overlap[j-i, m, n, 0]>0.99:
							t_angle = angle[j-i, m, n]
							t_angle = (np.where(t_angle==np.max(t_angle))[0][0] + 0.5)*np.pi/30
							t_size = size[j-i, m, n]*142
							center_x = t_size[0] + self.grid[m,n,0]
							center_y = t_size[1] + self.grid[m,n,1]
							t_half_width = t_size[2] / 2
							t_half_height = t_size[3] / 2
							pnts = [
								[center_x-t_half_width*np.cos(t_angle)+t_half_height*np.sin(t_angle),center_y-t_half_width*np.sin(t_angle)-t_half_height*np.cos(t_angle)],
								[center_x+t_half_width*np.cos(t_angle)+t_half_height*np.sin(t_angle),center_y+t_half_width*np.sin(t_angle)-t_half_height*np.cos(t_angle)],
								[center_x+t_half_width*np.cos(t_angle)-t_half_height*np.sin(t_angle),center_y+t_half_width*np.sin(t_angle)+t_half_height*np.cos(t_angle)],
								[center_x-t_half_width*np.cos(t_angle)-t_half_height*np.sin(t_angle),center_y-t_half_width*np.sin(t_angle)+t_half_height*np.cos(t_angle)]]
							pnts = np.int32([pnts])
							img = cv2.polylines(img, pnts, True, (255,0,0), 1)
				cv2.imwrite(output_folder + test_files[j], img)
				print(output_folder + test_files[j])
		session.close()
	
	def test_size_angle_predictor(self, test_folder='./test/', output_folder='./output/', batch_size=10, model_path='./model/model'):
		test_files = os.listdir(test_folder)
		X = tf.placeholder(tf.float32, shape=[None, 480, 640, 3])
		F = self.extract_features(X, training=False, padding='same', add_noise=False)
		PY1 = self.predict_overlap(F, padding='same')
		PY2 = self.predict_size(F, padding='same')
		PY3 = self.predict_angle(F, padding='same', training=False)
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)
		n_data = len(test_files)
		for i in range(0, n_data, batch_size):
			imgs = []
			end_i = min(i+batch_size, n_data)
			for j in range(i, end_i):
				imgs.append(cv2.imread(test_folder + test_files[j]))
			imgs = np.float32(imgs)
			overlap, size, angle = session.run([PY1, PY2, PY3], feed_dict={X: imgs/255})
			for j in range(i, end_i):
				img = imgs[j-i]
				for m in range(30):
					for n in range(40):
						t_overlap = overlap[j-i,m,n]
						t_size = size[j-i,m,n]*142
						if t_overlap[0]>0.99 and 55<=t_size[0]<=87 and 55<=t_size[1]<=87:
							print(t_size)
							img = cv2.rectangle(img, (self.grid[m,n,0],self.grid[m,n,1]), (self.grid[m,n,2],self.grid[m,n,3]), (255,0,0), 1)
				cv2.imwrite(output_folder + test_files[j], img)
				print(output_folder + test_files[j])
		session.close()
	
model = Model(
	background_folder= base_folder + 'background/')

'''
model.train(
	image_folder = base_folder + 'full_dataset/train_image/', 
	label_folder = base_folder + 'full_dataset/train_label/rotated_rectangle/', 
	n_epoch = 100, 
	batch_size = 15, 
	model_path = base_folder + 'model/model', 
	resume = False)
'''

model.test(
	test_folder = base_folder + 'sub_dataset/test_image/',
	output_folder = base_folder + 'sub_dataset/temp_output/',
	batch_size = 10,
	model_path = base_folder + 'model/model')