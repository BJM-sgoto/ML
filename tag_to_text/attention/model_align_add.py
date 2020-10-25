# reference : 
# https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f
# https://blog.floydhub.com/attention-mechanism/
# https://github.com/topics/local-attention
# local attention implementation: https://github.com/gugundi/NeuralMachineTranslation/blob/master/model.py
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf

BATCH_SIZE = 70
RANDOM_SEED = None
ENCODER_DIM = 256
DECODER_DIM = 256
SCORE_HIDDEN_DIM = 256
DICTIONARY_DIM = 32
CHAR_START = '$'
CHAR_END = '&'
CHAR_PAD = '#'
TEXT_LEN = 20

np.random.seed(RANDOM_SEED)
tf.reset_default_graph()
tf.disable_v2_behavior()
tf.set_random_seed(RANDOM_SEED)


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
			text_len = TEXT_LEN
			char_ids = list(np.random.randint(low=0, high=dict_size, size=[text_len]))
			text = [self.chars[i] for i in char_ids]
			text = ''.join(text)
			#print('text', text)
			
			background_color = np.random.uniform(low=0, high=255.0, size=[3]) #background_color = np.zeros([3], dtype=np.float32)
			
			text_thickness = np.random.choice([1,2])
			
			text_color = np.random.uniform(low=0, high=255.0, size=[3])
			#text_color = np.ones([3], dtype=np.float64) * 255
			
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
				
		max_width = int(np.ceil((max_width - 46)/32)*32+46)
		images = [self.pad_image(image, max_width) for image in images]	
		images = np.float32(images)
		
		texts = [self.pad_text(text, max_len) for text in texts]
		texts = [[self.dictionary[c] for c in text] for text in texts]
		texts = np.int32(texts)
		
		return images, texts
			
	def pad_image(self, image, new_width):
		height, width, _ = image.shape
		background_color = np.random.uniform(low=0, high=255.0, size=[3])
		new_image = np.ones([height, new_width, 3], dtype=np.float32) * background_color
		paste_x = np.random.randint(low=0, high=new_width + 1 - width)
		new_image[:, paste_x:paste_x + width] = image
		return new_image
	
	def pad_text(self, text, new_len):
		text = CHAR_START + text + CHAR_END + CHAR_PAD * (new_len - len(text))
		return text
	
	def encode(self, images, training=False):
		with tf.variable_scope('encoder'):
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
				features = tf.layers.batch_normalization(features, training=training)
			features = tf.squeeze(features, axis=1) # batch, times,  features
		
		with tf.variable_scope('encoder'):
			encoder_cell = tf.nn.rnn_cell.GRUCell(ENCODER_DIM, name='encoder_cell')
			encoder_output, _ = tf.nn.dynamic_rnn(
				encoder_cell,
				features,
				dtype=tf.float32,
				time_major=False)
		
		return encoder_output
		
	def attention(self, decoder_hidden, encoder_output):
		with tf.variable_scope('attention'):
			decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, axis=1)
			score_part1 = tf.layers.dense(
				decoder_hidden_with_time_axis,
				units=SCORE_HIDDEN_DIM)
			score_part2 = tf.layers.dense(
				encoder_output,
				units=SCORE_HIDDEN_DIM)
			score = tf.nn.tanh(score_part1 + score_part2)
			score = tf.layers.dense(
				score,
				units=1)
			attention_weights = tf.nn.softmax(score, axis=1)
			context_vector = encoder_output * attention_weights
			context_vector = tf.reduce_sum(context_vector, axis=1)
			return context_vector
	
	def decode(self, encoder_output, decoder_input_texts, decoder_hidden):
		decoder_dictionary = np.eye(len(self.dictionary), dtype=np.float32)
		with tf.variable_scope('decoder'):
			context_vector = self.attention(decoder_hidden, encoder_output)
			# decode
			'''
			decoder_dictionary = tf.get_variable(
				'decoder_dictionary',
				shape=[len(self.dictionary), DICTIONARY_DIM],
				dtype=tf.float32)
			'''
			decoder_input = tf.gather_nd(decoder_dictionary, tf.expand_dims(decoder_input_texts, axis=2))
			decoder_input = tf.concat([decoder_input, tf.expand_dims(context_vector, axis=1)], axis=-1)
			#
			decoder_cell = tf.nn.rnn_cell.GRUCell(ENCODER_DIM, name='decoder_cell')
			decoder_output, decoder_state = tf.nn.dynamic_rnn(
				decoder_cell,
				decoder_input,
				dtype=tf.float32,
				time_major=False)
			decoder_output = tf.layers.dense(
				decoder_output,
				units=len(self.dictionary))
			decoder_output = tf.nn.softmax(decoder_output, axis=-1)
			return decoder_output, decoder_state

	def train_on_batch(self, session, tf_train_op, tf_loss, tf_precision, tf_encoder_input_texts, tf_decoder_input_texts, tf_decoder_hidden_states, tf_decoder_output_texts, tf_mask, tf_predicted_decoder_texts, tf_predicted_decoder_next_hidden_state):
		n_samples = BATCH_SIZE
		encoder_input_texts, output_texts = self.make_batch(n_samples=n_samples)
		decoder_hidden_states = np.zeros([n_samples, DECODER_DIM], dtype=np.float32)
		text_len = len(output_texts[0]) - 1
		mean_loss_val = 0.0
		mean_precision_val = 0.0
		for i in range(text_len):	
			decoder_input_texts = output_texts[:,i: i+1]
			decoder_output_texts = output_texts[:,i+1: i+2]
			mask = np.float32(np.where(decoder_output_texts==self.dictionary[CHAR_PAD], 0.0, 1.0))
			
			_, loss_val, precision_val, predicted_decoder_texts, decoder_hidden_states = session.run(
				[tf_train_op, 
				tf_loss, 
				tf_precision,
				tf_predicted_decoder_texts,
				tf_predicted_decoder_next_hidden_state],
				feed_dict={
					tf_encoder_input_texts: encoder_input_texts,
					tf_decoder_input_texts: decoder_input_texts,
					tf_decoder_output_texts: decoder_output_texts,
					tf_decoder_hidden_states: decoder_hidden_states,
					tf_mask: mask})
			mean_loss_val += loss_val
			mean_precision_val += precision_val
		mean_loss_val = mean_loss_val / text_len
		mean_precision_val = mean_precision_val / text_len
		return mean_loss_val, mean_precision_val
			
	def train(self, n_loop=1000, model_path='./model/model', resume=False):
		X = tf.placeholder(tf.float32, [None, 46, None, 3])
		F = self.encode(X, training=True)
		
		Y = tf.placeholder(tf.int32, [None, 1])
		H = tf.placeholder(tf.float32, [None, DECODER_DIM])
		P_NY, P_NH = self.decode(F, Y, H) # predicted next y, next hidden state
		mask = tf.placeholder(tf.float32, [None, 1])
		NY = tf.placeholder(tf.int32, [None, 1])
		labels = tf.one_hot(NY, depth=len(self.dictionary))
		loss = tf.reduce_mean(tf.square(P_NY - labels), axis=2)
		loss = loss * mask
		loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
		
		optimizer = tf.train.AdamOptimizer(5e-4)
		train_op = optimizer.minimize(loss)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		
		grads_vars = optimizer.compute_gradients(loss)
		grads = [grad_var[0] for grad_var in grads_vars]
		grad_placeholders = [(tf.placeholder(tf.float32, shape=grad_var[0].get_shape()), grad_var[1]) for grad_var in grads_vars]
		apply_grad_op = optimizer.apply_gradients(grad_placeholders)
		
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
			loss_val, precision_val = self.train_on_batch(session, train_op, loss, precision, X, Y, H, NY, mask, P_NY, P_NH)
			mean_loss = (mean_loss * count_to_save + loss_val)/ (count_to_save + 1)
			print('Loop {:02d} Loss {:06f} Mean Loss {:06f} Precesion {:06f}'.format(i,loss_val, mean_loss, precision_val))
			count_to_save+=1
			if count_to_save>=100:
				count_to_save = 0
				mean_loss = 0
				print('---------------------------\nSave model')
				saver.save(session, model_path)
		session.close()
		
	
	def test(self, model_path='./model/model'):
		n_samples = BATCH_SIZE
		X = tf.placeholder(tf.float32, [None, 46, None, 3])
		F = self.encode(X)
		
		Y = tf.placeholder(tf.int32, [None, 1])
		H = tf.placeholder(tf.float32, [None, DECODER_DIM])
		P_NY, P_NH = self.decode(F, Y, H) # predicted next y, next hidden state
		
		encoder_input_texts, output_texts = self.make_batch(n_samples=n_samples)
		decoder_hidden_states = np.zeros([n_samples, DECODER_DIM], dtype=np.float32)
		text_len = len(output_texts[0]) - 1
		
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		outputs = []
		decoder_input_texts = np.ones([BATCH_SIZE, 1], dtype=np.float32) * self.dictionary[CHAR_START]
		for i in range(text_len):	
			predicted_decoder_texts, decoder_hidden_states = session.run(
				[P_NY,
				P_NH],
				feed_dict={
					X: encoder_input_texts,
					Y: decoder_input_texts,
					H: decoder_hidden_states})
			decoder_input_texts = np.argmax(predicted_decoder_texts, axis=2)
			outputs.append(decoder_input_texts)
		
		outputs = np.concatenate(outputs, axis=1)
		inputs = output_texts[:, 1:]
		mask = np.where(np.equal(inputs,self.dictionary[CHAR_PAD]), 0.0, 1.0)
		print('Input\n', inputs)
		print('Outputs\n', outputs)
		precision = np.where(np.equal(inputs,outputs), 1.0, 0.0) * mask
		print('Precesion', precision)
		print('Precesion', np.sum(precision)/ np.sum(mask))
		session.close()
		
model = Model()
model.train(n_loop=5000, model_path='./model_align_add/model_20', resume=False)
#model.test(model_path='./model_align_add/model_20')