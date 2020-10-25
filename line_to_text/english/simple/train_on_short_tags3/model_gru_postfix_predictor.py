# this model can regconize the rest of text  if we pass 2 char
# ex: image : "abcdefg", input text: "cd" => output text: "efg"

import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random

RANDOM_SEED = None
BATCH_SIZE = 50

#CUT_WIDTH = 46 + 3 * 16
#CUT_HEIGHT = 46
#STRIDE = 8 * 16
CUT_WIDTH = 36+8*11
CUT_HEIGHT = 36
STRIDE = 8*6 # OVERLAP = 48

ENCODER_DIM = 256
DECODER_DIM = 256
COMBINER_DIM = 256
COMBINER_DICTIONARY_DIM = 32
DECODER_DICTIONARY_DIM = 32
NUM_INPUT_TEXT = 2

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
			accumulated_text_len = eval(s[2])
			dataset.append([image_path, encoded_text, accumulated_text_len])			
			s = f.readline()
		f.close()
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	# batch item [image, accumulated_text_lens, text]
	def pad_image(self, image, new_height, new_width):
		image_height, image_width, _ = image.shape
		# cut 2 ends, pad upper and lower sides
		cut_left = np.random.randint(low = 7, high = image_width + 1 - new_width) # 7 ~ min width of chars
		pad_up = np.random.randint(low = 0, high = new_height + 1 - image_height)
		new_image = np.ones([new_height, new_width, 3], dtype=np.float32) * image[0,0] # background color
		new_image[pad_up : pad_up + image_height] = image[:, cut_left : cut_left + new_width]
		return new_image

	def pad_text(self, text, new_len):
		# only pad right with end char
		return text + [self.dictionary[CHAR_PAD]] * (new_len - len(text))
	
	def process_item(self, image_path, text, accumulated_text_len):
		image = np.float32(cv2.imread(image_path))
		image_height = image.shape[0]
		first_char_width = accumulated_text_len[0]
		text_width = accumulated_text_len[-1]
		cut_left = min(int(first_char_width/2), text_width - CUT_WIDTH - STRIDE)
		new_image = np.ones([CUT_HEIGHT, CUT_WIDTH + STRIDE, 3], dtype=np.float32) * image[0,0]
		pad_up = np.random.randint(low = 0, high = CUT_HEIGHT + 1 - image_height)
		new_image[pad_up : pad_up + image_height] = image[:, cut_left : cut_left + CUT_WIDTH + STRIDE]
		start_pos=0
		for start_pos in range(len(text)):
			if accumulated_text_len[start_pos]>cut_left:
				break
		end_pos = len(text)-1
		for end_pos in range(len(text)-1, -1, -1):
			if accumulated_text_len[end_pos] - cut_left <= CUT_WIDTH + STRIDE:
				break
		new_text = text[start_pos: end_pos]
		return new_image, new_text
	
	def process_items(self, batch):
		new_images, new_texts = [], []
		for image_path, text, accumulated_text_len in batch:
			new_image, new_text = self.process_item(image_path, text, accumulated_text_len)
			new_images.append(new_image)
			new_text = [self.dictionary[CHAR_START]] + new_text + [self.dictionary[CHAR_END]]
			new_texts.append(new_text)
		max_text_len = max(len(new_text) for new_text in new_texts)
		new_texts = [self.pad_text(new_text, max_text_len) for new_text in new_texts]
		new_images = np.float32(new_images)
		new_texts = np.int32(new_texts)
		return new_images, new_texts
	
	def encode(self, image_input, training=False):
		with tf.variable_scope('encoder', reuse=self.reuse_encoder):
			if training:
				noise = tf.random.uniform(tf.shape(image_input), minval=-5.0, maxval=5.0, dtype=tf.float32)
				image_input = image_input + noise
			features = image_input / 255
			layer_depths = [64,128,256]
			for layer_depth in layer_depths:
				features = tf.layers.conv2d(
					features,
					filters=layer_depth,
					kernel_size=(3,3),
					strides=(1,1),
					padding='valid')
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
			print('---------------------\nFeature', features)
			features = tf.squeeze(features, axis=1)
			encoder_cell = tf.nn.rnn_cell.GRUCell(ENCODER_DIM, name='encoder_cell')
			_, features = tf.nn.dynamic_rnn(
				encoder_cell,
				features,
				dtype=tf.float32,
				time_major=False)
			
			self.reuse_encoder = True
			return features
	
	def combine(self, feature1, feature2, decoder_init_text_inputs):
		with tf.variable_scope('combiner', reuse=self.reuse_combiner):
			combiner_dictionary = tf.get_variable('combiner_dictionary', shape=[len(self.dictionary), COMBINER_DICTIONARY_DIM], dtype=tf.float32)
			decoder_init_input = tf.gather_nd(combiner_dictionary, tf.expand_dims(decoder_init_text_inputs, axis=2))
			decoder_init_input_shape = tf.shape(decoder_init_input)
			decoder_init_input = tf.reshape(decoder_init_input, [decoder_init_input_shape[0], NUM_INPUT_TEXT * len(self.dictionary)])
			context = tf.concat([feature1, feature2, decoder_init_input], axis=1)
			context = tf.layers.dense(
				context,
				units=COMBINER_DIM)
			self.reuse_combiner = True
			return context
		
	def decode(self, context_vector, decoder_text_inputs):
		with tf.variable_scope('decoder', reuse=self.reuse_decoder):
			decoder_dictionary = tf.get_variable('decoder_dictionary', shape=[len(self.dictionary), DECODER_DICTIONARY_DIM], dtype=tf.float32)
			decoder_input = tf.gather_nd(decoder_dictionary, tf.expand_dims(decoder_text_inputs, axis=2))
			decoder_cell = tf.nn.rnn_cell.GRUCell(DECODER_DIM, name='decoder_cell')
			decoder_output, decoder_state = tf.nn.dynamic_rnn(
				decoder_cell,
				decoder_input,
				initial_state=context_vector,
				time_major=False)
			
			decoder_text_output = tf.layers.dense(
				decoder_output,
				units=len(self.dictionary))
			decoder_text_output = tf.nn.softmax(decoder_text_output, axis=-1)
			
			self.reuse_decoder = True
			return decoder_text_output, decoder_state
	
	def train_on_batch(self, batch, session, tf_encoder_input1, tf_encoder_input2, tf_decoder_init_input, tf_decoder_input, tf_target_output, tf_mask, tf_train_op, tf_loss, tf_precision, tf_decoder_output):
		num_samples = len(batch)
		images, texts = self.process_items(batch)
		images1 = images[:, :CUT_WIDTH]
		images2 = images[:, STRIDE:]
		mean_loss_val = 0
		mean_precision_val = 0
		sum_sum_mask = 0
		text_len = texts.shape[1]
		for i in range(text_len-NUM_INPUT_TEXT):
			init_texts = texts[:, i:i+NUM_INPUT_TEXT]
			input_texts = texts[:,i+NUM_INPUT_TEXT-1:-1]
			target_texts = texts[:, i+NUM_INPUT_TEXT:]
			mask = np.float32(np.where(np.equal(target_texts, self.dictionary[CHAR_PAD]), 0, 1))
			_, loss_val, precision_val, pred_text = session.run(
				[tf_train_op,
				tf_loss,
				tf_precision,
				tf_decoder_output],
				feed_dict={
					tf_encoder_input1: images1,
					tf_encoder_input2: images2,
					tf_decoder_init_input: init_texts,
					tf_decoder_input: input_texts,
					tf_target_output: target_texts,
					tf_mask: mask})
			sum_mask = np.sum(mask)
			mean_loss_val += loss_val * sum_mask
			mean_precision_val += precision_val * sum_mask
			sum_sum_mask += sum_mask
		mean_loss_val /= sum_sum_mask
		mean_precision_val /= sum_sum_mask
		return mean_loss_val, mean_precision_val
		
	def train(self, num_epochs=1000, batch_size=10, image_folder='./image/', dataset_file='./dataset.txt', model_path='./model/model', resume=False):
		X1 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX1 = self.encode(X1, training=True)
		X2 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX2 = self.encode(X2, training=True)
		init_Y = tf.placeholder(tf.int32, shape=[None, 2])
		print('-------------------------\n', FX1, FX2)
		C = self.combine(FX1, FX2, init_Y)
		
		prev_Y = tf.placeholder(tf.int32, shape=[None, None])
		pred_Y, _ = self.decode(C, prev_Y)
		mask = tf.placeholder(tf.float32, shape=[None, None])
		
		Y = tf.placeholder(tf.int32, shape=[None, None])
		labels = tf.one_hot(Y, depth=len(self.dictionary))
		loss = tf.reduce_mean(tf.square(labels - pred_Y), axis=2)
		loss = loss * mask
		loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
		
		optimizer = tf.train.AdamOptimizer(5e-4)
		train_op = optimizer.minimize(loss)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		
		precision = tf.where(tf.equal(Y, tf.argmax(pred_Y, axis=2, output_type=tf.int32)), tf.ones_like(mask), tf.zeros_like(mask))
		precision = tf.reduce_sum(precision * mask)/ tf.reduce_sum(mask)
		
		session = tf.Session()
		saver = tf.train.Saver()
		
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
		
				loss_val, precision_val = self.train_on_batch(batch, session, X1, X2, init_Y, prev_Y, Y, mask, train_op, loss, precision, pred_Y)
		
				print('Epoch {:02d} Progress {:05d} Loss {:06f} Precision {:06f}'.format(i, j , loss_val,  precision_val))
				count_to_save+=1
				if count_to_save>=100:
					print('-------------------\nSave\n-------------------')
					count_to_save = 0
					saver.save(session, model_path)
		session.close()
		
		
	def test(self, image_folder='./dataset/', dataset_file='./dataset.txt', model_path='./model/model'):
		X1 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX1 = self.encode(X1, training=False)
		X2 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX2 = self.encode(X2, training=False)
		init_Y = tf.placeholder(tf.int32, shape=[None, 2])
		pred_C = self.combine(FX1, FX2, init_Y)
		
		C = tf.placeholder(tf.float32, shape=[None, COMBINER_DIM])
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
			images = [self.pad_image(image, CUT_HEIGHT, CUT_WIDTH + STRIDE)[0] for image in images]
			images = np.float32(images)
			images1 = images[:, :, : CUT_WIDTH]
			images2 = images[:, :, STRIDE:]
			texts = [item[1] for item in batch]
			init_text = np.float32(np.ones([end_i-i, NUM_INPUT_TEXT]) * self.dictionary[CHAR_START])
			#init_text[:, 0] = self.dictionary['o']
			#init_text[:, 1] = self.dictionary['s']
			context = session.run(pred_C, feed_dict={X1: images1, X2: images2, init_Y: init_text})
		
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
				pos = decoder_pred_text.find(CHAR_END)
				if pos!=-1:
					decoder_pred_text = decoder_pred_text[:pos]
				decoder_target_text = decoder_target_texts[k-i][-len(decoder_pred_text):]
				equal = decoder_target_text == decoder_pred_text
				file_name = batch[k-i][0]
				print(file_name, decoder_target_text, '--->', decoder_pred_text, ':', equal)
				if equal:
					precision += 1
		#precision = precision / n_data
		print('Precision = {:d}/{:d} {:04f}'.format(precision, n_data, precision/n_data))
		session.close()
		
	def test_on_long_text(self, image_path='./test_image.png', model_path='./model/model'):
		X1 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX1 = self.encode(X1, training=False)
		X2 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX2 = self.encode(X2, training=False)
		init_Y = tf.placeholder(tf.int32, shape=[None, 2])
		pred_C = self.combine(FX1, FX2, init_Y)
		
		C = tf.placeholder(tf.float32, shape=[None, COMBINER_DIM])
		prev_Y = tf.placeholder(tf.int32, shape=[None, None])
		pred_Y, decoder_S = self.decode(C, prev_Y)
		
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		
		reversed_dictionary = {}
		for item in self.dictionary.items():
			reversed_dictionary[item[1]] = item[0]
		
		image = np.float32(cv2.imread(image_path))
		print('---------------\nImage shape', image.shape)
		image_height, image_width, _ = image.shape
		n = int((image_width - CUT_WIDTH)/STRIDE) + 1
		image_width = n * STRIDE + CUT_WIDTH
		image_height = CUT_HEIGHT
		image, _ = self.pad_image(image, image_height, image_width)
		reversed_dictionary = {}
		for item in self.dictionary.items():
			reversed_dictionary[item[1]] = item[0]
		
		init_texts = np.ones([1, NUM_INPUT_TEXT], dtype=np.int32) * self.dictionary[CHAR_START]
		text_inputs = np.ones([1,1], dtype=np.int32)*self.dictionary[CHAR_START]	
		num_samples = 1
		pred_texts = [[] for j in range(num_samples)]
		for i in range(n):
			dones = np.zeros(num_samples, dtype=np.int32)
			image1 = np.float32(image[:, i*STRIDE:i*STRIDE+CUT_WIDTH])
			image2 = np.float32(image[:, (i+1)*STRIDE:(i+1)*STRIDE+CUT_WIDTH])
			image3 = np.float32(image[:, i*STRIDE:(i+1)*STRIDE+CUT_WIDTH])
			cv2.imwrite('image1_{:06d}.png'.format(i), image1)
			cv2.imwrite('image3_{:06d}.png'.format(i), image3)
			image1 = np.expand_dims(image1, axis=0)
			image2 = np.expand_dims(image2, axis=0)
			contexts = session.run(pred_C, feed_dict={X1: image1, X2: image2, init_Y: init_texts})
			while True:
				text_outputs, new_contexts = session.run([pred_Y, decoder_S], feed_dict={C: contexts, prev_Y: text_inputs})
				text_outputs = np.argmax(text_outputs, axis=2)
				for j in range(num_samples):
					pred_texts[j].append(reversed_dictionary[text_outputs[j,0]])
					if not dones[j] and text_outputs[j,0]==self.dictionary[CHAR_END]:
						pred_texts[j].pop(-1)
						dones[j]=1
				if np.sum(dones)==num_samples:
					break
						
				contexts = new_contexts
				text_inputs = text_outputs
			# recompute init texts
			for j in range(num_samples):
				pred_text = pred_texts[j]
				for k in range(min(len(pred_text), NUM_INPUT_TEXT) - 1, -1, -1):
					init_texts[j, NUM_INPUT_TEXT-1-k] = self.dictionary[pred_text[-1-k]]
			
			for j in range(num_samples):
				print(''.join(pred_texts[j]))
	
	def test_on_long_text2(self, batch_size=10, image_folder='./image/', model_path='./model/model'):
		X1 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX1 = self.encode(X1, training=False)
		X2 = tf.placeholder(tf.float32, shape=[None, CUT_HEIGHT, CUT_WIDTH, 3])
		FX2 = self.encode(X2, training=False)
		init_Y = tf.placeholder(tf.int32, shape=[None, 2])
		pred_C = self.combine(FX1, FX2, init_Y)
		
		C = tf.placeholder(tf.float32, shape=[None, COMBINER_DIM])
		prev_Y = tf.placeholder(tf.int32, shape=[None, None])
		pred_Y, decoder_S = self.decode(C, prev_Y)
		
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		
		reversed_dictionary = {}
		for item in self.dictionary.items():
			reversed_dictionary[item[1]] = item[0]
		
		image_files = os.listdir(image_folder)
		num_data = len(image_files)
		for b in range(0, num_data, batch_size):
			end_b = min(b+batch_size, num_data)
			num_samples = end_b - b
			images = [np.float32(cv2.imread(image_folder + image_files[i])) for i in range(b, end_b)]
			image_width = max(image.shape[1] for image in images)
			image_height = max(image.shape[0] for image in images)
		
			n = int((image_width - CUT_WIDTH)/STRIDE) + 1
			image_width = n * STRIDE + CUT_WIDTH
			image_height = CUT_HEIGHT
			images = [self.pad_image(image, image_height, image_width)[0] for image in images]
			images = np.float32(images)
			reversed_dictionary = {}
			for item in self.dictionary.items():
				reversed_dictionary[item[1]] = item[0]
			
			init_texts = np.ones([num_samples, NUM_INPUT_TEXT], dtype=np.int32) * self.dictionary[CHAR_START]
			text_inputs = np.ones([num_samples,1], dtype=np.int32)*self.dictionary[CHAR_START]	
			pred_texts = [[] for j in range(num_samples)]
			for i in range(n):
				dones = np.zeros(num_samples, dtype=np.int32)
				image1 = images[:, :, i*STRIDE:i*STRIDE+CUT_WIDTH]
				image2 = images[:, :, (i+1)*STRIDE:(i+1)*STRIDE+CUT_WIDTH]
				
				contexts = session.run(pred_C, feed_dict={X1: image1, X2: image2, init_Y: init_texts})
				while True:
					text_outputs, new_contexts = session.run([pred_Y, decoder_S], feed_dict={C: contexts, prev_Y: text_inputs})
					text_outputs = np.argmax(text_outputs, axis=2)
					for j in range(num_samples):
						if text_outputs[j,0]!=self.dictionary[CHAR_END]:
							pred_texts[j].append(reversed_dictionary[text_outputs[j,0]])
						elif dones[j]==0:
							dones[j]=1
					if np.sum(dones)==num_samples:
						break
							
					contexts = new_contexts
					text_inputs = text_outputs
				# recompute init texts
				for j in range(num_samples):
					pred_text = pred_texts[j]
					for k in range(min(len(pred_text), NUM_INPUT_TEXT) - 1, -1, -1):
						init_texts[j, NUM_INPUT_TEXT-1-k] = self.dictionary[pred_text[-1-k]]
				
			for j in range(num_samples):
				print(''.join(pred_texts[j]))
	
model = Model()
model.train(num_epochs=150, batch_size=BATCH_SIZE, image_folder='./train_dataset/', dataset_file='./train_img_text.txt', model_path='./model_gru_postfix_predictor/model', resume=False)
#model.test(image_folder='./temp_test/', dataset_file='./temp_test_img_text.txt',  model_path='./model_gru_postfix_predictor/model')
#model.test_on_long_text(image_path='./test_dataset_long_text/00000.png', model_path='./model_gru_postfix_predictor/model')
