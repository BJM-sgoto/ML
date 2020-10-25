import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random

RANDOM_SEED = 1234
BATCH_SIZE = 80

CUT_WIDTH = 46+48
CUT_HEIGHT = 46
ENCODER_DIM = 256
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
		self.reuse_encoder = False
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
			dataset.append([image_path, encoded_text])
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
		
	def find_right_text(self, accumulated_text_len):
		num_chars = len(accumulated_text_len)
		end_char_id = 0
		for i in range(num_chars):
			if accumulated_text_len[i]>CUT_WIDTH:
				end_char_id = i
				break
		else:
			end_char_id = num_chars
		end_char_id += 1
		return end_char_id
		
	def encode(self, image_input, training=False):
		with tf.variable_scope('encoder', reuse=self.reuse_encoder):
			if training:
				noise = tf.random.uniform(tf.shape(image_input), minval=-5.0, maxval=5.0, dtype=tf.float32)
				image_input = image_input + noise
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
			
			features = tf.squeeze(features, axis=1)
			encoder_cell = tf.nn.rnn_cell.GRUCell(ENCODER_DIM, name='encoder_cell')
			_, features = tf.nn.dynamic_rnn(
				encoder_cell,
				features,
				dtype=tf.float32,
				time_major=False)
			
			self.reuse_encoder = True
			return features
		
	def decode(self, context_vector, decoder_text_input):
		decoder_dictionary = np.eye(len(self.dictionary), dtype=np.float32) # 
		with tf.variable_scope('decoder', reuse=self.reuse_decoder):
			decoder_text_input = tf.gather_nd(decoder_dictionary, tf.expand_dims(decoder_text_input, axis=2))
			decoder_cell = tf.nn.rnn_cell.GRUCell(DECODER_DIM, name='decoder_cell')
			decoder_output, decoder_state = tf.nn.dynamic_rnn(
				decoder_cell,
				decoder_text_input,
				initial_state=context_vector,
				time_major=False)
			decoder_text_output = tf.layers.dense(
				decoder_output,
				units=len(self.dictionary))
			decoder_text_output = tf.nn.softmax(decoder_text_output, axis=-1)
			self.reuse_decoder = True
			return decoder_text_output, decoder_state
	
	def train_on_batch(self, batch, session, tf_encoder_input, tf_decoder_input, tf_target_output, tf_mask, tf_train_op, tf_loss, tf_precision):
		num_samples = len(batch)
		images = []
		texts = []
		accumulated_text_lens = []
		for item in batch:
			image = np.float32(cv2.imread(item[0]))
			image, _ = self.pad_image(image, CUT_HEIGHT, CUT_WIDTH)
			images.append(image)
			texts.append([self.dictionary[CHAR_START]] + item[1] + [self.dictionary[CHAR_END]])
		max_len = max(len(text) for text in texts)
		texts = [self.pad_text(text, max_len) for text in texts]
		
		mean_precision = 0.0
		mean_loss = 0.0
		
		encoder_input = np.float32(images)
		texts = np.int32(texts)
		decoder_input = texts[:, :-1]
		target_output = texts[:, 1:]
		mask = np.float32(np.where(np.equal(target_output, self.dictionary[CHAR_PAD]), 0, 1))
	
		_, loss_val, precision_val = session.run(
			[tf_train_op,
			tf_loss,
			tf_precision],
			feed_dict={
				tf_encoder_input: encoder_input,
				tf_decoder_input: decoder_input,
				tf_target_output: target_output,
				tf_mask: mask})
					
		return loss_val, precision_val		
		
	def train(self, num_epochs=1000, batch_size=10, image_folder='./image/', dataset_file='./dataset.txt', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX = self.encode(X, training=True)
		
		prev_Y = tf.placeholder(tf.int32, shape=[None, None])
		pred_Y, _ = self.decode(FX, prev_Y)
		
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
				loss_val, precision_val = self.train_on_batch(batch, session, X, prev_Y, Y, mask, train_op, loss, precision)
				print('Epoch', i, 'Progress', j, 'Loss', loss_val, 'Precision', precision_val)
				count_to_save+=1
				if count_to_save>=100:
					print('-------------------\nSave\n-------------------')
					count_to_save = 0
					saver.save(session, model_path)
		session.close()
	
	def test(self, image_folder='./dataset/', dataset_file='./dataset.txt', model_path='./model/model'):
		X = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX = self.encode(X, training=False)
		
		C = tf.placeholder(tf.float32, [None, ENCODER_DIM])
		prev_Y = tf.placeholder(tf.int32, shape=[None, None])
		pred_Y, decoder_S = self.decode(C, prev_Y)
		
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		
		reversed_dictionary = {}
		for item in self.dictionary.items():
			reversed_dictionary[item[1]] = item[0]
		
		test_dataset = self.make_dataset(image_folder=image_folder, dataset_file=dataset_file)
		n_data = len(test_dataset)
		
		precision = 0
		
		for i in range(0, n_data, BATCH_SIZE):
			end_i = min(i + BATCH_SIZE, n_data)
			batch = test_dataset[i: end_i]
			images = [cv2.imread(item[0]) for item in batch]
			texts = [item[1] for item in batch]
			images = [self.pad_image(image, CUT_HEIGHT, CUT_WIDTH)[0] for image in images]
			images = np.float32(images)
			context = session.run(FX, feed_dict={X: images})
			prev_text = np.float32(np.ones([end_i-i, 1]) * self.dictionary[CHAR_START]) 
			pred_texts = []
			max_len = max(len(text) for text in texts) + 1
			for j in range(max_len-1):
				pred_text, context = session.run([pred_Y, decoder_S], feed_dict={C: context, prev_Y: prev_text})
				prev_text = np.argmax(pred_text, axis=2)
				pred_texts.append(prev_text)
				
			pred_texts = np.concatenate(pred_texts, axis=1)
			
			decoder_pred_texts = ['' for k in range(i, end_i)]
			for j in range(max_len - 1):
				for k in range(i, end_i):
					decoder_pred_texts[k-i] += reversed_dictionary[pred_texts[k - i][j]]
			
			decoder_target_texts = [''.join(reversed_dictionary[c] for c in text) for text in texts]
			#print(decoder_pred_texts)
			#print(decoder_target_texts)
			
			for k in range(i, end_i):
				decoder_pred_text = decoder_pred_texts[k-i]
				decoder_target_text = decoder_target_texts[k-i]
				pos = decoder_pred_text.find(CHAR_END)
				if pos!=-1:
					decoder_pred_text = decoder_pred_text[:pos]
				equal = decoder_target_text == decoder_pred_text
				file_name = batch[k-i][0]
				print(file_name, decoder_target_text, '--->', decoder_pred_text, ':', equal)
				if equal:
					precision += 1
		#precision = precision / n_data
		print('Precision = {:d}/{:d} {:04f}'.format(precision, n_data, precision/n_data))
		session.close()				
				
				
model = Model()
#model.train(num_epochs=150, batch_size=BATCH_SIZE, image_folder='./train_dataset/', dataset_file='./train_img_text.txt', model_path='./model_gru_predictor/model', resume=True)
model.test(image_folder='./test_dataset/', dataset_file='./test_img_text.txt',  model_path='./model_gru_predictor/model')