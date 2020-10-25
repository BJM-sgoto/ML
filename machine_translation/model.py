# reference: https://github.com/tensorflow/nmt
import tensorflow.compat.v1 as tf
import numpy as np
import random
import re

tf.disable_v2_behavior()
tf.reset_default_graph()

PAD_START = '<start>'
PAD_END = '<end>'
class Model:
	def __init__(self, data_file='./spa_eng.txt', source_dict_file='./dict/source_dict.txt', target_dict_file='./dict/target_dict.txt'):
		self.data_file = data_file
		self.source_dict_file = source_dict_file
		self.target_dict_file = target_dict_file
				
	def write_dict(self, dict_source_word2index, dict_target_word2index, source_dict_file='./dict/source_dict.txt', target_dict_file='./dict/target_dict.txt'):
		s = ''
		f = open(source_dict_file, 'w')
		for word, index in dict_source_word2index.items():
			s += word + str(index) + '\n'
		f.write(s)
		f.close()
		
		s = ''
		f = open(target_dict_file, 'w')
		for word, index in dict_target_word2index.items():
			s += word + ' ' + str(index) + '\n'
		f.write(s)
		f.close()
	
	def read_dict(self, source_dict_file='./dict/source_dict.txt', target_dict_file='./dict/target_dict.txt'):
		f = open(source_dict_file, 'r')
		dict_source_word2index = {}
		dict_source_index2word = {}
		while True:
			s = f.readline()
			word, index = s.strip().split(' ')
			index = int(index)
			dict_source_word2index[word] = index
			dict_source_index2word[index] = word
			if not s:
				break
		f.close()
		
		f = open(target_dict_file, 'r')
		dict_target_word2index = {}
		dict_target_index2word = {}
		while True:
			s = f.readline()
			word, index = s.strip().split(' ')
			index = int(index)
			dict_target_word2index[word] = index
			dict_target_index2word[index] = word
			if not s:
				break
		f.close()
		dict_source_word2index, dict_source_index2word, dict_target_word2index, dict_target_index2word
		
	def preprocess_sentence(self, s):
		# unicode to ascii ??
		# lower
		s = s.lower().strip()
		# pad space between words and punctuations
		s = re.sub(r"([?.!,¿])", r" \1 ", s)
		s = re.sub(r'[" "]+', " ", s)
		# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "¿")
		s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
		# adding a start and an end token to the sentence
		# so that the model know when to start and stop predicting.
		s = PAD_START + ' ' + s + ' ' + PAD_END
		# convert to list of words
		s = s.split(' ')
		s = [word for word in s if len(word)>0]
		return s
		
	def make_raw_dataset(self, data_file='./spa_end.txt'):
		f = open(data_file, 'r', encoding='utf-8')
		ss = f.read().strip()
		f.close()
		dataset = [[self.preprocess_sentence(sub_s) for sub_s in s.split('\t')] for s in ss.split('\n')]
		return dataset
		
	def make_lookup_dicts(self, raw_dataset):
		dict_source_word2index = {}
		dict_source_index2word = {}
		dict_target_word2index = {}
		dict_target_index2word = {}
		source_word_count = 0
		target_word_count = 0
		for source_sentence, target_sentence in raw_dataset:
			for source_word in source_sentence:
				if source_word not in dict_source_word2index:
					dict_source_word2index[source_word] = source_word_count
					source_word_count += 1
			for target_word in target_sentence:
				if target_word not in dict_target_word2index:
					dict_target_word2index[target_word] = target_word_count
					target_word_count +=1
		for source_word in dict_source_word2index:
			dict_source_index2word[dict_source_word2index[source_word]] = source_word
		for target_word in dict_target_word2index:
			dict_target_index2word[dict_target_word2index[target_word]] = target_word
		return dict_source_word2index, dict_source_index2word, dict_target_word2index, dict_target_index2word
		
	def make_index_dataset(self, raw_dataset, dict_source_word2index, dict_target_word2index):
		index_dataset = []
		for source_sentence, target_sentence in raw_dataset:
			source_index_sentence = [dict_source_word2index[w] for w in source_sentence]
			target_index_sentence = [dict_target_word2index[w] for w in target_sentence]
			index_dataset.append([source_index_sentence, target_index_sentence])
		return index_dataset
		
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
		
	def init_model(self, encoder_inputs, decoder_inputs, source_vocab_size, target_vocab_size, source_code_len=300, target_code_len=300, encoder_dim=1024, decoder_dim=1024):
		# define embedding rules
		embedded_source_dict = tf.get_variable(name='embedded_source_dict', shape=[source_vocab_size, source_code_len])
		embedded_target_dict = tf.get_variable(name='embedded_target_dict', shape=[target_vocab_size, target_code_len])
		
		encoder_inputs = tf.expand_dims(encoder_inputs, axis=2) # shape [batch_size=None, sentence_len=None, index=1]
		embedded_encoder_inputs = tf.gather_nd(embedded_source_dict, encoder_inputs)
		# encode input to context vector
		encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_dim, name='encoder_cell')
		encoder_outputs, (encoder_state_h, encoder_state_c) = tf.nn.dynamic_rnn(
			encoder_cell,
			embedded_encoder_inputs,
			dtype=tf.float32,
			time_major=False) # time_major=True-> first dimension is batch
		
		# convert encoder_state to decoder_outputs
		decoder_inputs = tf.expand_dims(decoder_inputs, axis=2)
		embedded_decoder_inputs = tf.gather_nd(embedded_target_dict, decoder_inputs)
		
		decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_dim, name='decoder_cell')
		decoder_outputs, _ = tf.nn.dynamic_rnn(
			decoder_cell,
			embedded_decoder_inputs,
			#initial_state=(encoder_state_h, encoder_state_c),
			dtype=tf.float32,
			time_major=False)
		
		# decode output to words
		decoder_outputs = tf.layers.dense(
			decoder_outputs,
			units=target_vocab_size)
		return decoder_outputs
	
	def pad_index_sentence(self, index_sentences, dict_word2index):
		longest_sentence = max(index_sentences, key=lambda index_sentence: len(index_sentence))
		max_len = len(longest_sentence)
		index_end = dict_word2index[PAD_END]
		# TODO HERE
	
	def train(self, num_epoch=100, model_path='./model/model', resume=False):
		if resume:
			raw_dataset = self.make_raw_dataset(self.data_file)
			dict_source_word2index, dict_source_index2word, dict_target_word2index, dict_target_index2word = self.make_lookup_dicts(raw_dataset)
			self.write_dict(dict_source_word2index, dict_target_word2index, source_dict_file='./dict/source_dict.txt', target_dict_file='./dict/target_dict.txt')
			dataset = self.make_index_dataset(raw_dataset, dict_source_word2index, dict_target_word2index)
		else:
			raw_dataset = self.make_raw_dataset(self.data_file)
			dict_source_word2index, dict_source_index2word, dict_target_word2index, dict_target_index2word = self.read_dict(source_dict_file='./dict/source_dict.txt', target_dict_file='./dict/target_dict.txt')
			dataset = self.make_index_dataset(raw_dataset, dict_source_word2index, dict_target_word2index)
			
		X_S = tf.placeholder(tf.int32, [None, None])
		X_T = tf.placeholder(tf.int32, [None, None])
		source_vocab_size=len(dict_source_word2index)
		target_vocab_size=len(dict_target_word2index)
		PY_T = self.init_model(X_S, X_T, source_vocab_size, target_vocab_size)
		Y_T = tf.placeholder(tf.int32, [None, None])
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_T, logits=PY_T)
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
			saver.
		else:
			session.run(tf.global_variables_initializer())
		
		
		
model = Model(data_file='./spa_eng.txt')
model.train()
