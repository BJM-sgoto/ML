#https://mlexplained.com/2018/04/29/paper-dissected-glove-global-vectors-for-word-representation-explained/
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
tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)

ENCODE_DIM = 200
VOCAB_SIZE = 10000

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

	def make_cooccurence_mat(self, encoded_dataset, window_size=3, first_time=True):
		if first_time:
			cooccurence_mat = np.zeros([VOCAB_SIZE, VOCAB_SIZE], dtype=np.float32)
			for i, sentence in enumerate(encoded_dataset):
				print(i)
				sentence_len = len(sentence)
				for word_pos in range(sentence_len):
					word = sentence[word_pos]
					min_neighbor_pos = max(0, word_pos - window_size)
					max_neighbor_pos = min(sentence_len, word_pos + window_size)
					for neighbor_pos in range(min_neighbor_pos, max_neighbor_pos):
						neighbor = sentence[neighbor_pos]
						cooccurence_mat[word, neighbor]+=1
						cooccurence_mat[neighbor, word]+=1
			np.save('cooccurence_mat.npy', cooccurence_mat)
		else:
			cooccurence_mat = np.load('cooccurence_mat.npy')
		return cooccurence_mat

	def build_model(self, encoded_input1, encoded_input2):
		with tf.variable_scope("Glove"):
			encoded_vocab = tf.get_variable("encoded_vocab", dtype=tf.float32, shape=[VOCAB_SIZE, ENCODE_DIM])
			biases = tf.get_variable("biases", dtype=tf.float32, shape=[ENCODE_DIM])
			encoded_input1 = tf.expand_dims(encoded_input1, axis=1)
			encoded_input2 = tf.expand_dims(encoded_input2, axis=1)
			x1 = tf.gather_nd(encoded_vocab, encoded_input1)
			x2 = tf.gather_nd(encoded_vocab, encoded_input2)
			bias1 = tf.gather_nd(biases, encoded_input1)
			bias2 = tf.gather_nd(biases, encoded_input2)
			y = tf.reduce_sum(x1 * x2, axis = 1)
			y = y + bias1 + bias2
		return y, encoded_vocab

	def sample(self, cooccurence_mat, n_samples):
		inputs1 = np.random.randint(0, VOCAB_SIZE, size=n_samples)
		inputs2 = np.random.randint(0, VOCAB_SIZE, size=n_samples)
		outputs = cooccurence_mat[inputs1, inputs2]
		return inputs1, inputs2, outputs

	def train(self, train_file, test_file, processed_file, vocab_file, first_time=True, n_steps=50000, batch_size=512,model_path='./model/cbow', resume=False):

		dataset = self.make_dataset(train_file, test_file, processed_file, first_time=first_time)
		word2index, index2word = self.make_vocab(dataset, vocab_file, vocab_size=VOCAB_SIZE, first_time=first_time)
		dataset = self.encode_dataset(dataset, word2index)
		cooccurence_mat = self.make_cooccurence_mat(dataset, first_time=first_time)

		X1 = tf.placeholder(tf.int32, shape=[None])
		X2 = tf.placeholder(tf.int32, shape=[None])
		Y = tf.placeholder(tf.float32, shape=[None])
		PY, encoded_vocab = self.build_model(X1, X2)
		loss = tf.square(PY - tf.log(1+Y))
		Y_100 = Y / 100
		W = tf.where(tf.greater(Y_100, 1), tf.ones_like(Y_100), Y_100)
		loss = tf.reduce_sum(W * loss)
		train_op = tf.train.AdamOptimizer().minimize(loss)
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		for i in range(n_steps):
			inputs1, inputs2, outputs = self.sample(cooccurence_mat, n_samples=512)
			loss_val, _ = session.run([loss, train_op], feed_dict={X1: inputs1, X2: inputs2, Y:outputs})
			print("Step {:06d}, Loss {:06f}".format(i, loss_val))
			if (i+1)%1000==0:
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
	model_path='./model/glove', 
	resume=False)