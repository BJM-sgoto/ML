# train_file : "id","qid1","qid2","question1","question2","is_duplicate"
# test_file : "test_id","question1","question2"
# contractions: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
# https://arxiv.org/pdf/1301.3781.pdf
# https://medium.com/analytics-vidhya/implementing-word2vec-in-tensorflow-44f93cf2665f
import re
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)

ENCODE_DIM = 200
VOCAB_SIZE = 10000
MIN_LEN = 5 # min num of context words on 1 side to predict a word
MAX_LEN = 10 # max num of context words on 1 side to predict a word
START_SUB_SENTENCE_ID = VOCAB_SIZE
END_SUB_SENTENCE_ID = VOCAB_SIZE + 1

class Model:
	def __init__(self):
		pass

	def make_dataset(self, train_file, test_file, processed_file, first_time=True):
		dataset = []
		if first_time:
			train_data = pd.read_csv(train_file)
			train_data["question1"] = train_data["question1"].astype(str)
			train_data["question2"] = train_data["question2"].astype(str)
			dataset += list(train_data["question1"].values)
			dataset += list(train_data["question2"].values)
			test_data = pd.read_csv(test_file)
			test_data["question1"] = test_data["question1"].astype(str)
			test_data["question2"] = test_data["question2"].astype(str)
			dataset += list(test_data["question1"].values)
			dataset += list(test_data["question2"].values)
			dataset = self._preprocess(dataset)
			f = open(processed_file, "w")
			for item in dataset:
				f.write(item + "\n")
			f.close()
		else:
			f = open(processed_file, "r")
			s = f.readline().strip()
			while s:
				dataset.append(s)
				s = f.readline().strip()
			f.close()
		return dataset

	def _preprocess(self, dataset):
		# lower text
		dataset = [item.lower() for item in dataset]

		# replace contractions
		contractions = { 
			"ain't": "am not",
			"aren't": "are not",
			"can't": "cannot",
			"can't've": "cannot have",
			"'cause": "because",
			"could've": "could have",
			"couldn't": "could not",
			"couldn't've": "could not have",
			"didn't": "did not",
			"doesn't": "does not",
			"don't": "do not",
			"hadn't": "had not",
			"hadn't've": "had not have",
			"hasn't": "has not",
			"haven't": "have not",
			"she'd": "she would",
			"she'd've": "she would have",
			"she'll": "she will",
			"she'll've": "she will have",
			"she's": "she is",
			"he'd": "he would",
			"he'd've": "he would have",
			"he'll": "he will",
			"he'll've": "he will have",
			"he's": "he is",
			"how'd": "how did",
			"how'd'y": "how do you",
			"how'll": "how will",
			"how's": "how is",
			"I'd": "I would",
			"I'd've": "I would have",
			"I'll": "I will",
			"I'll've": "I will have",
			"I'm": "I am",
			"I've": "I have",
			"isn't": "is not",
			"it'd": "it would",
			"it'd've": "it would have",
			"it'll": "it will",
			"it'll've": "it will have",
			"it's": "it is",
			"let's": "let us",
			"ma'am": "madam",
			"mayn't": "may not",
			"might've": "might have",
			"mightn't": "might not",
			"mightn't've": "might not have",
			"must've": "must have",
			"mustn't": "must not",
			"mustn't've": "must not have",
			"needn't": "need not",
			"needn't've": "need not have",
			"o'clock": "of the clock",
			"oughtn't": "ought not",
			"oughtn't've": "ought not have",
			"shan't": "shall not",
			"sha'n't": "shall not",
			"shan't've": "shall not have",
			"should've": "should have",
			"shouldn't": "should not",
			"shouldn't've": "should not have",
			"so've": "so have",
			"so's": "so is",
			"that'd": "that had",
			"that'd've": "that would have",
			"that's": "that is",
			"there'd": "there would",
			"there'd've": "there would have",
			"there's": "there is",
			"they'd": "they would",
			"they'd've": "they would have",
			"they'll": "they will",
			"they'll've": "they will have",
			"they're": "they are",
			"they've": "they have",
			"to've": "to have",
			"wasn't": "was not",
			"we'd": "we would",
			"we'd've": "we would have",
			"we'll": "we will",
			"we'll've": "we will have",
			"we're": "we are",
			"we've": "we have",
			"weren't": "were not",
			"what'll": "what will",
			"what'll've": "what will have",
			"what're": "what are",
			"what's": "what is",
			"what've": "what have",
			"when's": "when is",
			"when've": "when have",
			"where'd": "where did",
			"where's": "where is",
			"where've": "where have",
			"who'll": "who will",
			"who'll've": "who will have",
			"who's": "who is",
			"who've": "who have",
			"why's": "why is",
			"why've": "why have",
			"will've": "will have",
			"won't": "will not",
			"won't've": "will not have",
			"would've": "would have",
			"wouldn't": "would not",
			"wouldn't've": "would not have",
			"y'all": "you all",
			"y'all'd": "you all would",
			"y'all'd've": "you all would have",
			"y'all're": "you all are",
			"y'all've": "you all have",
			"you'd": "you would",
			"you'd've": "you would have",
			"you'll": "you will",
			"you'll've": "you will have",
			"you're": "you are",
			"you've": "you have"
		}
		for key in contractions.keys():
			print("Replacing", key)
			dataset = [item.replace(key.lower(), contractions[key].lower()) for item in dataset]

		# replace numbers with #
		number_pattern = r'[-+]?([0-9]*\.[0-9]+|[0-9]+)'
		dataset = [re.sub(number_pattern, "<number>", item) for item in dataset]

		# remove , : ; ' " + -  * / ! ?
		dataset = [re.sub(r'[,:;\'"\+\-\*\\/\!\?]', " ", item) for item in dataset]

		# remove \n
		dataset = [item.replace("\n", " ") for item in dataset]
		
		# add <start> and <end> to sentences
		dataset = ["<start> " + item + " <end>" for item in dataset]
		
		# combine multiple spaces
		dataset = [re.sub(r' +', " ", item) for item in dataset]
		
		return dataset

	def make_vocab(self, processed_dataset, vocab_file, vocab_size=VOCAB_SIZE, first_time=True):
		word2index = {}
		index2word = {}
		if first_time:
			word_counts = {}
			for item in processed_dataset:
				words = item.split(" ")
				for word in words:
					if word in word_counts:
						word_counts[word] += 1
					else:
						word_counts[word] = 1
			word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)	
			f = open(vocab_file, "w")
			for count in range(vocab_size):
				word = word_counts[count][0]
				index2word[count] = word
				word2index[word] = count
				f.write(word + "\n")
			f.close()
		else:
			f = open(vocab_file, "r")
			word = f.readline().strip()
			count = 0
			while word:
				word2index[word] = count
				index2word[count] = word
				count+=1
				word = f.readline().strip()
			f.close()
		return word2index, index2word

	def encode_dataset(self, dataset, word2index):
		encoded_dataset = []
		for item in dataset:
			words = item.split(" ")
			words = [word2index[word] for word in words if word in word2index]
			encoded_dataset.append(words)
		return encoded_dataset

	def build_model(self, left_encoded_input, right_encoded_input):
		with tf.variable_scope("OOV"):
			encoded_vocab = tf.get_variable("encoded_vocab", dtype=tf.float32, shape=[VOCAB_SIZE+2, ENCODE_DIM]) # 2 paddings at start and end of sub sentences
			x1 = tf.expand_dims(left_encoded_input, axis=2)
			x1 = tf.gather_nd(encoded_vocab, x1)
			left_gru_cell = tf.nn.rnn_cell.GRUCell(num_units=200, name="left_cell")
			_, x1 = tf.nn.dynamic_rnn(
				left_gru_cell,
				x1,
				dtype=tf.float32)
			x2 = tf.expand_dims(right_encoded_input, axis=2)
			x2 = tf.gather_nd(encoded_vocab, x2)
			right_gru_cell = tf.nn.rnn_cell.GRUCell(num_units=200, name="right_cell")
			_, x2 = tf.nn.dynamic_rnn(
				right_gru_cell,
				x2,
				dtype=tf.float32)
			y = tf.concat([x1,x2], axis=1)
			y = tf.layers.dense(y, units=200)
			y = tf.layers.dense(y, units=VOCAB_SIZE)
			y = tf.nn.softmax(y, axis=1)
		return y, encoded_vocab

	def sample(self, dataset, n_samples):
		sentence_ids = np.random.randint(0, len(dataset), n_samples)
		word_ids = np.random.randint(0, 1000, n_samples)
		left_neighbors = []
		right_neighbors = []
		outputs = []
		for i in range(n_samples):
			sentence = dataset[sentence_ids[i]]
			sentence_len = len(sentence)
			word_id = word_ids[i]%sentence_len
			left, right = np.random.randint(MIN_LEN, MAX_LEN+1, size=2)
			min_neighbor_id = max(0, word_id - left)
			max_neighbor_id = min(sentence_len, word_id + right + 1)
			left_neighbors.append(sentence[min_neighbor_id:word_id])
			right_neighbors.append(sentence[word_id+1:max_neighbor_id])
			outputs.append(sentence[word_id])
		max_left_neighbor = len(max(left_neighbors, key=lambda x: len(x)))
		max_right_neighbor = len(max(right_neighbors, key=lambda x: len(x)))
		left_neighbors = [[START_SUB_SENTENCE_ID] * (max_left_neighbor - len(neighbor)) + neighbor for neighbor in left_neighbors]
		right_neighbors = [neighbor + [END_SUB_SENTENCE_ID] * (max_right_neighbor - len(neighbor)) for neighbor in right_neighbors]
		left_neighbors = np.int32(left_neighbors)
		right_neighbors = np.int32(right_neighbors)
		outputs = np.int32(outputs)
		return left_neighbors, right_neighbors, outputs

	def train(self, train_file, test_file, processed_file, vocab_file, first_time=True, n_steps=50000, batch_size=512,model_path='./model/oov', resume=False):
		dataset = self.make_dataset(train_file, test_file, processed_file, first_time=first_time)
		word2index, index2word = self.make_vocab(dataset, vocab_file, vocab_size=VOCAB_SIZE, first_time=first_time)
		dataset = self.encode_dataset(dataset, word2index)
		X1 = tf.placeholder(tf.int32, shape=[None, 10])
		X2 = tf.placeholder(tf.int32, shape=[None, 10]) # TODO
		Y = tf.placeholder(tf.int32, shape=[None])
		
		PY, encoded_vocab = self.build_model(X1, X2)
			
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(PY - tf.one_hot(Y, depth=VOCAB_SIZE, dtype=tf.float32)), axis=1))
		train_op = tf.train.AdamOptimizer().minimize(loss)
		saver = tf.train.Saver()
		session = tf.Session()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		for i in range(n_steps):
			left_neighbors, right_neighbors, outputs = self.sample(dataset, batch_size)
			loss_val, _ = session.run([loss, train_op], feed_dict={X1: left_neighbors, X2: right_neighbors, Y: outputs})
			print("Step {:06d}, Loss {:06f}".format(i, loss_val))
			if (i+1)%5000==0:
				saver.save(session, model_path)
		saver.save(session, model_path)

		session.close()
		

model = Model()
model.train(
	train_file="../train.csv",
	test_file="../train.csv",
	processed_file="../processed_data.txt",
	vocab_file="../vocab.txt",
	first_time=False,
	n_steps=50000, 
	batch_size=1024,
	model_path='./model/oov_v1', 
	resume=True)