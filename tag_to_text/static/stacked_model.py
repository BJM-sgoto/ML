# https://colah.github.io/posts/2015-08-Understanding-LSTMs/

import cv2
import random

import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf

BATCH_SIZE = 20
RANDOM_SEED = None
ENCODER_DIM = 512
DECODER_DIM = 512
CHAR_START = '$'
CHAR_END = '&'
CHAR_PAD = '#'
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.disable_v2_behavior()
tf.set_random_seed(RANDOM_SEED)
tf.reset_default_graph()

class Model:
	def __init__(self):
		self.dictionary = {}
		for i in range(ord('a'), ord('z') + 1):
			self.dictionary[chr(i)] = i - ord('a')
		self.dictionary[CHAR_START] = len(self.dictionary)
		self.dictionary[CHAR_END] = len(self.dictionary)
		self.dictionary[CHAR_PAD] = len(self.dictionary)
		
	def make_dataset(self, image_folder='./train_image/', dataset_file='./dataset.txt'):
		f = open(dataset_file, 'r')
		dataset = []
		s = f.readline()
		while s:
			s = s.strip().split('\t')
			dataset.append([image_folder + s[0], CHAR_START + s[1] + CHAR_END])
			s = f.readline()
		f.close()
		return dataset
		
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def train_on_batch(self, session, tf_train_op, tf_loss, tf_precision, tf_input_images, tf_input_texts, tf_output_texts, tf_mask, batch, tf_output):
		images, texts = self.process_batch(batch)
		input_texts = texts[:, :-1]
		output_texts = texts[:, 1:]
		mask = np.float32(np.where(output_texts==self.dictionary[CHAR_PAD], 0.0, 1.0))
		_, loss_val, precision_val, predicted_output = session.run([tf_train_op, tf_loss, tf_precision, tf_output], 
			feed_dict={
				tf_input_images: images,
				tf_input_texts: input_texts,
				tf_output_texts: output_texts,
				tf_mask: mask})
		#print('input_texts', input_texts)
		#print('output_texts', output_texts)
		#print('predicted_output', np.argmax(predicted_output, axis=2))
		#exit()
		return loss_val, precision_val
		
	def process_batch(self, raw_batch):
		images = []
		max_width = 0
		
		texts = []
		max_len = 0
		for item in raw_batch:
			# image
			image = cv2.imread(item[0])
			images.append(image)
			width = image.shape[1]
			if max_width < width:
				max_width = width
			
			# text
			text = item[1]
			texts.append(text)
			if max_len < len(text):
				max_len = len(text)
		max_width = int((max_width - 46)//32 + 1) * 32 + 46
		images = [self.pad_image(image, max_width) for image in images]	
		images = np.float32(images)
		texts = [self.pad_text(text, max_len) for text in texts]
		texts = [[self.dictionary[c] for c in text] for text in texts]
		texts = np.int32(texts)
		return images, texts
	
	def pad_image(self, image, new_width):
		height, width, _ = image.shape
		#background_color = np.random.uniform(low=0, high=255.0, size=[3])
		background_color = np.zeros([3], dtype=np.float32)
		new_image = np.ones([height, new_width, 3], dtype=np.float32) * background_color
		paste_x = np.random.randint(low=0, high=new_width + 1 - width)
		new_image[:, paste_x:paste_x + width] = image
		return new_image
	
	def pad_text(self, text, new_len):
		text = text + CHAR_PAD * (new_len - len(text))
		return text
	
	def encode(self, images, training=False):
		with tf.variable_scope('extractor'):
			features = images / 255
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
				#F = tf.layers.batch_normalization(F, training=training)
			features = tf.squeeze(features, axis=1) # batch, times,  features
			
		with tf.variable_scope('encoder_1'):
			encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(ENCODER_DIM, name='encoder_cell_1')
			Fs, features = tf.nn.dynamic_rnn(
				encoder_cell,
				features,
				dtype=tf.float32,
				time_major=False)
				
		with tf.variable_scope('encoder_2'):
			encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(ENCODER_DIM, name='encoder_cell_2')
			_, encoder_output = tf.nn.dynamic_rnn(
				encoder_cell,
				Fs,
				initial_state=features,
				time_major=False)
			
		return encoder_output
		
	def decode(self, initial_state, input_texts):
		with tf.variable_scope('decoder'):
			encoded_dictionary = tf.get_variable(
				'encoded_dictionary',
				shape=[len(self.dictionary), 256],
				dtype=tf.float32)
			encoded_input_texts = tf.gather_nd(encoded_dictionary, tf.expand_dims(input_texts, axis=2))
			decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(DECODER_DIM, name='decoder_cell')
			predicted_output_texts, states = tf.nn.dynamic_rnn(
				decoder_cell,
				encoded_input_texts,
				initial_state=initial_state,
				time_major=False)
			predicted_output_texts = tf.layers.dense(predicted_output_texts, units=len(self.dictionary))
			predicted_output_texts = tf.nn.softmax(predicted_output_texts, axis=-1)
		return predicted_output_texts, states
	
	def train(self, n_epoch=10, image_folder='./dataset/', dataset_file='./dataset.txt', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, [None, 46, None, 3])
		Y = tf.placeholder(tf.int32, [None, None])
		NY = tf.placeholder(tf.int32, [None, None])
		F = self.encode(X, training=True)
		P_NY, _ = self.decode(F, Y)
		mask = tf.placeholder(tf.float32, [None, None])
		
		#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=NY, logits=P_NY) * mask
		loss = tf.reduce_sum(tf.square(P_NY - tf.one_hot(NY, depth=len(self.dictionary))), axis=2)
		loss = loss * mask
		loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
		train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		precision = tf.where(tf.equal(NY, tf.argmax(P_NY, axis=2, output_type=tf.int32)), tf.ones_like(mask), tf.zeros_like(mask))
		precision = tf.reduce_sum(precision*mask)/tf.reduce_sum(mask)
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		dataset = self.make_dataset(image_folder=image_folder, dataset_file=dataset_file)
		n_data = len(dataset)
		count_to_save = 0
		mean_loss = 0
		for i in range(n_epoch):
			self.shuffle_dataset(dataset)
			for j in range(0, n_data, BATCH_SIZE):
				end_j = min(j + BATCH_SIZE, n_data)
				batch = dataset[j: end_j]
				loss_val, precision_val = self.train_on_batch(session, train_op, loss, precision, X, Y, NY, mask, batch, P_NY)
				mean_loss = (mean_loss * count_to_save + loss_val)/ (count_to_save + 1)
				print('Epoch {:02d} Progress {:04d} Loss {:06f} Mean Loss {:06f} Precision {:06f}'.format(i,j,loss_val, mean_loss, precision_val))
				#exit()
				count_to_save+=1
				if count_to_save>=100:
					count_to_save = 0
					mean_loss = 0
					print('---------------------------\nSave model')
					saver.save(session, model_path)
		session.close()
	
	# the result in test function is not the same because when call pad_image function, random function was used
	def test(self, image_folder='./dataset/', dataset_file='./dataset.txt', model_path='./stacked_model/model'):
		X = tf.placeholder(tf.float32, [None, 46, None, 3])
		Y = tf.placeholder(tf.int32, [None, None])
		# encode
		F = self.encode(X, training=False)
		# decode
		Ec = tf.placeholder(tf.float32, [None, ENCODER_DIM])
		Eh = tf.placeholder(tf.float32, [None, ENCODER_DIM])
		P_NY, (Dc, Dh) = self.decode(tf.nn.rnn_cell.LSTMStateTuple(c=Ec, h=Eh), Y)
		
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
			images, _ = self.process_batch(batch)
			output_texts = ['' for i in range(i, end_i)]
			# compute encoder output
			state_c_val, state_h_val = session.run(
				F,
				feed_dict={
					X: images})		
		
			# compute decoder output
			input_texts = np.zeros([end_i - i, 1], dtype=np.int32)
			input_texts[:, 0] = self.dictionary[CHAR_START]
			for j in range(10):
				char_prob, state_h_val, state_c_val = session.run(
					[P_NY, Dh, Dc],
					feed_dict={
						Eh: state_h_val,
						Ec: state_c_val,
						Y: input_texts
					})
				char_prob = char_prob[:,0]
				char_pos = np.argmax(char_prob, axis=1)
				input_texts[:,0] = char_pos
				#if char_pos == self.dictionary[CHAR_END]:
				#	break
				#	break
				for k in range(i, end_i):
					output_texts[k - i] += reversed_dictionary[char_pos[k - i]]
			for k in range(i, end_i):
				text = output_texts[k-i]
				pos = text.find(CHAR_END)
				if pos!=-1:
					text = text[:pos]
				target_text = batch[k-i][1][1:-1]
				equal = target_text == text
				file_name = batch[k-i][0]
				print(file_name, target_text, '--->', text, ':', equal)
				if equal:
					precision += 1
		precision = precision/n_data
		print('Precision:', precision)
		session.close()
			
model = Model()
model.train(n_epoch=10, image_folder='./train_dataset/', dataset_file='./train_img_text.txt', model_path='./stacked_model/model', resume=True)
#model.test(image_folder='./test_dataset/', dataset_file='./test_img_text.txt', model_path='./stacked_model/model')