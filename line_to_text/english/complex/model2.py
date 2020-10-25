import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random

RANDOM_SEED = 1234
BATCH_SIZE = 40

CUT_WIDTH = 142
CUT_HEIGHT = 46
STRIDE = 64
EXTRACTOR_OUTPUT_DIM = 256
DECODER_DIM = 256
COMBINER_DIM = 256

CHAR_START = '$'
CHAR_END = '&'
CHAR_PAD = '%'

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class Model:
	def __init__(self):
		self.reuse_extractor = False
		self.reuse_decoder = False
		self.reuse_combiner = False
		
		self.dictionary = {}
		for i in range(ord('a'), ord('z') + 1):
			self.dictionary[chr(i)] = i - ord('a')
		self.dictionary[CHAR_START] = len(self.dictionary)
		self.dictionary[CHAR_END] = len(self.dictionary)
		self.dictionary[CHAR_PAD] = len(self.dictionary)
	
	def make_dataset(self, image_folder='./image/', dataset_file='./dataset.txt'):
		f = open(dataset_file, 'r')
		dataset = []
		s = f.readline()
		while s:
			s = s.strip().split('\t')
			image_path = image_folder + s[0]
			text = s[1]
			encoded_text = [self.dictionary[c] for c in text]
			accumulated_text_lens = np.int32(eval(s[2]))
			dataset.append([image_path, encoded_text, accumulated_text_lens])
			s = f.readline()
		f.close()
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	# batch item [image, accumulated_text_lens, text]
	def pad_image(self, image, new_height, new_width):
		image_height, image_width, _ = image.shape
		pad_left = np.random.randint(low = 0, high = new_width + 1 - image_width)
		pad_up = np.random.randint(low = 0, high = new_height + 1 - image_height)
		new_image = np.ones([new_height, new_width, 3], dtype=np.float32) * image[0,0] # background color
		new_image[pad_up : pad_up + image_height, pad_left: pad_left+image_width] = image
		return new_image, pad_left

	def pad_text(self, text, new_len):
		# only pad right with end char
		return text + [self.dictionary[CHAR_PAD]] * (new_len - len(text))
		
	def find_text(self, accumulated_text_len, end, prev_end_id=0):
		num_chars = len(accumulated_text_len)
		end_char_id = prev_end_id
		for i in range(prev_end_id, num_chars):
			if accumulated_text_len[i]>end:
				end_char_id = i
				break
		else:
			end_char_id = num_chars
		return end_char_id
		
	def extract(self, image_input, training=False):
		with tf.variable_scope('extractor', reuse=self.reuse_extractor):
			features = image_input / 255
			layer_depths = [32,64,128,256]
			for layer_depth in layer_depths:
				features = tf.layers.conv2d(
					features,
					filters=layer_depth,
					kernel_size=(3,3),
					strides=(1,1),
					padding='valid',
					activation=tf.nn.elu)
				features = tf.layers.max_pooling2d(
					features,
					strides=(2,2),
					pool_size=(2,2))
				features = tf.layers.batch_normalization(features, training=training)
			features = tf.layers.flatten(features)
			features = tf.layers.dense(
				features,
				units=EXTRACTOR_OUTPUT_DIM)
			self.reuse_extractor = True
			return features
		
	def combine(self, prev_extractor_output, curr_extractor_output):
		with tf.variable_scope('combiner', reuse=self.reuse_combiner):
			context_vector = tf.concat([prev_extractor_output, curr_extractor_output],axis=1)
			#context_vector = tf.layers.dense(
			#	context_vector,
			#	units=COMBINER_DIM)
			return context_vector
		
	def decode(self, context_vector, decoder_text_input):
		decoder_dictionary = np.eye(len(self.dictionary), dtype=np.float32) # 
		with tf.variable_scope('decoder', reuse=self.reuse_decoder):
			decoder_text_input = tf.gather_nd(decoder_dictionary, tf.expand_dims(decoder_text_input, axis=2))
			decoder_cell = tf.nn.rnn_cell.GRUCell(DECODER_DIM, name='decoder_cell')
			decoder_output, decoder_state = tf.nn.dynamic_rnn(
				decoder_cell,
				decoder_text_input,
				dtype=tf.float32,
				time_major=False)
			decoder_text_output = tf.layers.dense(
				decoder_output,
				units=len(self.dictionary))
			decoder_text_output = tf.nn.softmax(decoder_text_output, axis=-1)
			self.reuse_decoder = True
			return decoder_text_output
	
	def train_on_batch(self, batch, session, tf_extractor_input1, tf_extractor_input2, tf_decoder_input, tf_target_output, tf_mask, tf_train_op, tf_loss, tf_precision):
		num_samples = len(batch)
		images = []
		texts = []
		accumulated_text_lens = []
		for item in batch:
			images.append(cv2.imread(item[0]))
			texts.append(item[1])
			accumulated_text_lens.append(item[2])
		new_width = max([image.shape[1] for image in images])
		num_parts = np.ceil((new_width - CUT_WIDTH) / STRIDE)
		new_width = int(num_parts * STRIDE + CUT_WIDTH)
		num_parts = int(num_parts + 1)
		for i, image in enumerate(images):
			images[i], pad_left = self.pad_image(image, CUT_HEIGHT, new_width)
			accumulated_text_lens[i] += pad_left
		
		# cut text to smaller parts
		sub_texts = [[] for j in range(num_parts)]
		prev_end_ids = [0 for i in range(num_samples)]
		for j in range(num_parts):
			for i in range(num_samples):
				end_id = self.find_text(accumulated_text_lens[i], j * STRIDE + CUT_WIDTH, prev_end_ids[i])
				sub_texts[j].append([self.dictionary[CHAR_START]] + texts[i][prev_end_ids[i]: end_id] + [self.dictionary[CHAR_END]])
				prev_end_ids[i] = end_id
		for j in range(num_parts):
			max_len = max(len(sub_text) for sub_text in sub_texts[j])
			for i, sub_text in enumerate(sub_texts[j]):
				sub_texts[j][i] = self.pad_text(sub_text, max_len)
		
		mean_precision = 0.0
		mean_loss = 0.0
		for j in range(num_parts):
			#for i in range(num_samples):
			extractor_input1 = []
			extractor_input2 = []
			decoder_input = []
			target_output = []
			for i in range(num_samples):
				if j==0:
					extractor_input1.append(np.float32(np.zeros([CUT_HEIGHT, CUT_WIDTH, 3])))
				else:
					extractor_input1.append(images[i][:, (j-1)*STRIDE: (j-1)*STRIDE + CUT_WIDTH])
				#print("extractor_input1", images[i][:, (j-1)*STRIDE: (j-1)*STRIDE + CUT_WIDTH].shape)
				extractor_input2.append(images[i][:, j*STRIDE: j*STRIDE + CUT_WIDTH])
				decoder_input.append(sub_texts[j][i][:-1])
				target_output.append(sub_texts[j][i][1:])
			extractor_input1 = np.float32(extractor_input1)
			extractor_input2 = np.float32(extractor_input2)
			decoder_input = np.int32(decoder_input)
			target_output = np.int32(target_output)
			"""
			if j==1:
				print(self.dictionary)
				print(decoder_input)
				print(target_output)
				for i in range(num_samples):
					cv2.imwrite('test{:06d}.bmp'.format(i), extractor_input[i])
				exit()
			"""
			mask = np.float32(np.where(np.equal(target_output, self.dictionary[CHAR_PAD]), 0, 1))
			
			_, loss_val, precision_val = session.run(
				[tf_train_op,
				tf_loss,
				tf_precision],
				feed_dict={
					tf_extractor_input1: extractor_input1,
					tf_extractor_input2: extractor_input2,
					tf_decoder_input: decoder_input,
					tf_target_output: target_output,
					tf_mask: mask})
			print('    Step {:02d}, Loss {:06f}, Precision {:06f}'.format(i, loss_val, precision_val))
			mean_loss += loss_val
			mean_precision += precision_val
		mean_loss/=num_parts
		mean_precision/=num_parts
		return mean_loss, mean_precision		
		
	def train(self, num_epochs=1000, batch_size=10, image_folder='./image/', dataset_file='./dataset.txt', model_path='./model/model', resume=False):
		X1 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX1 = self.extract(X1, training=True)
		X2 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX2 = self.extract(X2, training=True)
		
		C = self.combine(FX1, FX2)
		
		prev_Y = tf.placeholder(tf.int32, shape=[None, None])
		pred_Y = self.decode(C, prev_Y)
		
		mask = tf.placeholder(tf.float32, shape=[None, None])
		
		Y = tf.placeholder(tf.int32, shape=[None, None])
		labels = tf.one_hot(Y, depth=len(self.dictionary))
		loss = tf.reduce_mean(tf.square(labels - pred_Y), axis=2)
		loss = loss * mask
		loss = tf.reduce_sum(loss)/ tf.reduce_sum(mask)
		
		optimizer = tf.train.AdamOptimizer(5e-4)
		train_op = optimizer.minimize(loss)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		
		precision = tf.where(tf.equal(Y, tf.argmax(pred_Y, axis=2, output_type=tf.int32)), tf.ones_like(mask), tf.zeros_like(mask))
		precision = tf.reduce_sum(precision * mask)/ tf.reduce_sum(mask)
		
		session = tf.Session()
		saver = tf.train.Saver()
		for var in tf.trainable_variables():
			print(var)
		
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		dataset = self.make_dataset(image_folder=image_folder, dataset_file=dataset_file)
		num_data = len(dataset)
		count_to_save = 0
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j+batch_size)
				batch = dataset[j: end_j]
				loss_val, precision_val = self.train_on_batch(batch, session, X1, X2, prev_Y, Y, mask, train_op, loss, precision)
				print('Epoch', i, 'Progress', j, 'Loss', loss_val, 'Precision', precision_val)
				count_to_save+=1
				if count_to_save>=100:
					print('-------------------\nSave\n-------------------')
					saver.save(session, model_path)
		session.close()
				
				
				
model = Model()
model.train(
	num_epochs=1, 
	batch_size=BATCH_SIZE, 
	image_folder='./train_dataset/', 
	dataset_file='./train_img_text.txt',
	model_path='./model/model',
	resume=False)