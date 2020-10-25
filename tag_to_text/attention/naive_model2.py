# reference : 
# https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f
# https://blog.floydhub.com/attention-mechanism/
# https://stackoverflow.com/questions/34687761/efficiently-grab-gradients-from-tensorflow
# https://towardsdatascience.com/a-gentle-introduction-to-ocr-ee1469a201aa
import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf

BATCH_SIZE = 15
RANDOM_SEED = None
ENCODER_DIM = 256
DECODER_DIM = 256
SCORE_HIDDEN_DIM = 256
DICTIONARY_DIM = 128
CHAR_START = '$'
CHAR_END = '&'
CHAR_PAD = '#'

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
	
	def make_batch(self, n_samples=BATCH_SIZE):
		dict_size = len(self.dictionary) - 3 # do not use special chars
		inputs = []
		outputs = []
		
		for i in range(n_samples):
			text_len = np.random.randint(low=5, high=8)
			#text_len = 5
			char_ids = list(np.random.randint(low=0, high=dict_size, size=[text_len]))
			input_text = [self.chars[i] for i in char_ids]
			char_ids = list(np.random.randint(low=0, high=dict_size, size=[text_len]))
			output_text = input_text.copy()
			input_text = ''.join(input_text)
			output_text = ''.join(output_text)
			inputs.append(input_text)
			outputs.append(output_text)
			
		
		max_input_text_len = 0
		max_output_text_len = 0
		for input_text in inputs:
			if max_input_text_len < len(input_text):
				max_input_text_len = len(input_text)
		for output_text in outputs:
			if max_output_text_len < len(output_text):
				max_output_text_len = len(output_text)
		
		inputs = [self.pad_text(text, max_input_text_len) for text in inputs]
		outputs = [self.pad_text(text, max_output_text_len) for text in outputs]
		
		inputs = [[self.dictionary[c] for c in text] for text in inputs]
		inputs = np.int32(inputs)
		outputs = [[self.dictionary[c] for c in text] for text in outputs]
		outputs = np.int32(outputs)
		
		return inputs, outputs
			
	def pad_text(self, text, new_len):
		text = CHAR_START + text + CHAR_END + CHAR_PAD * (new_len - len(text))
		return text
	
	def encode(self, encoder_input_texts):
		with tf.variable_scope('encoder'):
			encoder_dictionary = tf.get_variable(
				'encoder_dictionary',
				shape=[len(self.dictionary), DICTIONARY_DIM],
				dtype=tf.float32)
			features = tf.gather_nd(encoder_dictionary, tf.expand_dims(encoder_input_texts, axis=2))
			# HERE
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
			score = tf.nn.tanh(score_part1 + score_part2) # TODO: concat
			score = tf.layers.dense(
				score,
				units=1)
			attention_weights = tf.nn.softmax(score, axis=1)
			context_vector = encoder_output * attention_weights
			context_vector = tf.reduce_sum(context_vector, axis=1)
			return context_vector
	
	#input_text size must be : None X 1 : only pass 1 word 
	def decode(self, encoder_output, decoder_input_texts, decoder_hidden):
		with tf.variable_scope('decoder'):
			context_vector = self.attention(decoder_hidden, encoder_output)
			# decode
			decoder_dictionary = tf.get_variable(
				'decoder_dictionary',
				shape=[len(self.dictionary), DICTIONARY_DIM],
				dtype=tf.float32)
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

	def train_on_batch(self, session, tf_gradients, tf_gradients_holders, tf_apply_grad_op, tf_loss, tf_precision, tf_encoder_input_texts, tf_decoder_input_texts, tf_decoder_hidden_states, tf_decoder_output_texts, tf_mask, tf_predicted_decoder_texts, tf_predicted_decoder_next_hidden_state):
		n_samples = BATCH_SIZE
		encoder_input_texts, output_texts = self.make_batch(n_samples=n_samples)
		decoder_hidden_states = np.zeros([n_samples, DECODER_DIM], dtype=np.float32)
		text_len = len(output_texts[0]) - 1
		mean_loss_val = 0.0
		mean_precision_val = 0.0
		mean_gradients = None
		for i in range(text_len):	
			decoder_input_texts = output_texts[:,i: i+1]
			decoder_output_texts = output_texts[:,i+1: i+2]
			mask = np.float32(np.where(decoder_output_texts==self.dictionary[CHAR_PAD], 0.0, 1.0))
			
			gradients_val, loss_val, precision_val, predicted_decoder_texts, decoder_hidden_states = session.run(
				[tf_gradients, 
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
			if mean_gradients is None:
				mean_gradients = gradients_val
			else:
				for j, gradient_val in enumerate(gradients_val):
					mean_gradients[j]+=gradient_val
			mean_loss_val += loss_val
			mean_precision_val += precision_val
		
		feed_dict = {}
		for j in range(len(mean_gradients)):
			mean_gradients[j]/=text_len
			feed_dict[tf_gradients_holders[j][0]] = mean_gradients[j]
		# update (change) variables
		session.run(tf_apply_grad_op, feed_dict=feed_dict)
		
		mean_loss_val = mean_loss_val / text_len
		mean_precision_val = mean_precision_val / text_len
		return mean_loss_val, mean_precision_val
			
	def train(self, n_loop=1000, model_path='./model/model', resume=False):
		X = tf.placeholder(tf.int32, [None, None])
		F = self.encode(X)
		
		Y = tf.placeholder(tf.int32, [None, 1])
		H = tf.placeholder(tf.float32, [None, DECODER_DIM])
		P_NY, P_NH = self.decode(F, Y, H) # predicted next y, next hidden state
		mask = tf.placeholder(tf.float32, [None, None])
		NY = tf.placeholder(tf.int32, [None, 1])
		labels = tf.one_hot(NY, depth=len(self.dictionary))
		loss = tf.reduce_mean(tf.square(P_NY - labels), axis=2)
		loss = loss * mask
		loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
		
		optimizer = tf.train.AdamOptimizer(5e-4)
		#train_op = optimizer.minimize(loss)
		#update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		#train_op = tf.group([train_op, update_op])
		
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
			loss_val, precision_val = self.train_on_batch(session, grads, grad_placeholders, apply_grad_op, loss, precision, X, Y, H, NY, mask, P_NY, P_NH)
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
		X = tf.placeholder(tf.int32, [None, None])
		F = self.encode(X)
		
		Y = tf.placeholder(tf.int32, [None, None])
		H = tf.placeholder(tf.float32, [None, DECODER_DIM])
		P_NY, P_NH = self.decode(F, Y, H) # predicted next y, next hidden state
		
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		mean_precision = 0.0
		for n in range(100):
			encoder_input_texts, output_texts = self.make_batch(n_samples=n_samples)
			decoder_hidden_states = np.zeros([n_samples, DECODER_DIM], dtype=np.float32)
			text_len = len(output_texts[0]) - 1
			
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
			inputs = encoder_input_texts[:, 1:]
			mask = np.where(np.equal(inputs,self.dictionary[CHAR_PAD]), 0.0, 1.0)
			#print('Input\n', inputs)
			#print('Outputs\n', outputs)
			precision = np.where(np.equal(inputs,outputs), 1.0, 0.0) * mask
			precision = np.sum(precision)/ np.sum(mask)
			print('Precesion', precision)
			mean_precision = (mean_precision * n + precision) / (n + 1)
		print('Mean Precesion', mean_precision)
		session.close()
		
model = Model()
#model.train(n_loop=5000, model_path='./model/model', resume=False)
model.test(model_path='./model/model')