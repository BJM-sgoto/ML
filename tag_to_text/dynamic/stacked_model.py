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
	def __init__(self, dict_file='./dict.txt'):
		self.chars = []
		self.dictionary = {}
		for i in range(ord('a'), ord('z') + 1):
			self.chars.append(chr(i))
			self.dictionary[chr(i)] = i - ord('a')
		self.dictionary[CHAR_START] = len(self.dictionary)
		self.dictionary[CHAR_END] = len(self.dictionary)
		self.dictionary[CHAR_PAD] = len(self.dictionary)
		print(self.dictionary)
		
		self.fonts = [cv2.FONT_HERSHEY_SIMPLEX, 
			cv2.FONT_HERSHEY_COMPLEX_SMALL,
			cv2.FONT_HERSHEY_DUPLEX,
			#cv2.FONT_HERSHEY_PLAIN,
			cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
			cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
			cv2.FONT_HERSHEY_SIMPLEX,
			cv2.FONT_HERSHEY_TRIPLEX,
			cv2.FONT_ITALIC]
			
		self.low_chars = ['p', 'q', 'g', 'y']
	
	def make_batch(self, n_samples=BATCH_SIZE):
		dict_size = len(self.dictionary) - 3 # do not use special chars
		font_scale = 1.0
		images = []
		texts = []
		
		for i in range(n_samples):
			#text_len = np.random.randint(low=10, high=16)
			text_len = 4
			char_ids = list(np.random.randint(low=0, high=dict_size, size=[text_len]))
			text = [self.chars[i] for i in char_ids]
			text = CHAR_START + ''.join(text) + CHAR_END
			#print('text', text)
			
			#background_color = np.random.uniform(low=0, high=255.0, size=[3]) # TODO : remove this
			background_color = np.zeros([3], dtype=np.float32)
			
			text_thickness = np.random.choice([1,2])
			#text_color = np.random.uniform(low=0, high=255.0, size=[3]) # TODO : remove this
			text_color = np.ones([3], dtype=np.float64) * 255
			
			font = np.random.choice(self.fonts)
			text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
			text_width, text_height = text_size
			img_height = text_height + 2 * text_thickness
			for low_char in self.low_chars:
				if low_char in text:
					img_height += baseline
					break
			else:
				baseline = 0
			img_width = text_width
			text_left = 0
			text_bottom = text_height + text_thickness
			
			padded_img_width = int((text_width - 46)//32 + 1) * 32 + 46
			padded_img_height = 46
			text_left = np.random.randint(low = 0, high = padded_img_width - img_width)
			text_bottom = np.random.randint(low = img_height - baseline, high = padded_img_height - baseline)
			img = np.ones([padded_img_height, padded_img_width, 3]) * background_color
			cv2.putText(img, text, (text_left, text_bottom), font, font_scale, text_color, text_thickness)
			images.append(img)
			texts.append(text)
		
		max_width = 0
		max_len = 0
		for i in range(n_samples):
			img_width = images[i].shape[1]
			if img_width > max_width:
				max_width = img_width
			
			text_len = len(texts[i])
			if text_len > max_len:
				max_len = text_len
				
		max_width = int((max_width - 46)//32 + 1) * 32 + 46
		images = [self.pad_image(image, max_width) for image in images]	
		images = np.float32(images)
		
		texts = [self.pad_text(text, max_len) for text in texts]
		texts = [[self.dictionary[c] for c in text] for text in texts]
		texts = np.int32(texts)
		for i, image in enumerate(images):
			cv2.imwrite('test{:06d}.bmp'.format(i), image)
		exit()
		return images, texts
			
	def train_on_batch(self, session, tf_train_op, tf_loss, tf_precision, tf_input_images, tf_input_texts, tf_output_texts, tf_mask):
		images, texts = self.make_batch()
		
		input_texts = texts[:, :-1]
		output_texts = texts[:, 1:]
		mask = np.float32(np.where(output_texts==self.dictionary[CHAR_PAD], 0.0, 1.0))
		_, loss_val, precision_val = session.run([tf_train_op, tf_loss, tf_precision], 
			feed_dict={
				tf_input_images: images,
				tf_input_texts: input_texts,
				tf_output_texts: output_texts,
				tf_mask: mask})
		return loss_val, precision_val
	
	def pad_image(self, image, new_width):
		height, width, _ = image.shape
		background_color = np.random.uniform(low=0, high=255.0, size=[3])
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
	
	def train(self, n_loop=10, image_folder='./dataset/', dataset_file='./dataset.txt', model_path='./model/model', resume=False):
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
		train_op = tf.group([train_op, update_op, P_NY])
		precision = tf.where(tf.equal(NY, tf.argmax(P_NY, axis=2, output_type=tf.int32)), tf.ones_like(mask), tf.zeros_like(mask))
		precision = tf.reduce_sum(precision*mask)/tf.reduce_sum(mask)
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		count_to_save = 0
		mean_loss = 0
		
		for i in range(n_loop):
			batch = self.make_batch(BATCH_SIZE)
			loss_val, precision_val = self.train_on_batch(session, train_op, loss, precision, X, Y, NY, mask)
			mean_loss = (mean_loss * count_to_save + loss_val)/ (count_to_save + 1)
			print('Loop {:02d} Loss {:06f} Mean Loss {:06f} Precesion {:06f}'.format(i,loss_val, mean_loss, precision_val))
			count_to_save+=1
			if count_to_save>=100:
				count_to_save = 0
				mean_loss = 0
				print('---------------------------\nSave model')
				saver.save(session, model_path)
		session.close()	
			
			
model = Model()
model.train(n_loop=5000, image_folder='./train_dataset/', dataset_file='./train_img_text.txt', model_path='./model/model', resume=False)