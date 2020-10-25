# https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f
# http://www.manythings.org/anki/

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense

import numpy as np
import pandas as pd

import string
import re

#base_folder = '/content/gdrive/My Drive/machine_learning_data/trans/'
base_folder = './'
lines = pd.read_table(base_folder + 'new_deu.txt', names=['eng', 'deu'])
batch_size = 32
latent_dim = 256
num_epoch = 10

lines.eng=lines.eng.apply(lambda x: x.lower())
lines.deu=lines.deu.apply(lambda x: x.lower())

# Remove all the special characters
lines.eng=lines.eng.apply(lambda x: re.sub('[' + string.punctuation + ']', ' ', x))
lines.deu=lines.deu.apply(lambda x: re.sub('[' + string.punctuation + ']', ' ', x))

# Remove extra spaces
lines.eng=lines.eng.apply(lambda x: x.strip())
lines.deu=lines.deu.apply(lambda x: x.strip())
lines.eng=lines.eng.apply(lambda x: re.sub(" +", " ", x))
lines.deu=lines.deu.apply(lambda x: re.sub(" +", " ", x))

# Add start and end tokens
lines.eng = lines.eng.apply(lambda x : 'START_ '+ x + ' _END')
lines.deu = lines.deu.apply(lambda x : 'START_ '+ x + ' _END')

# Vocabulary of English
all_eng_words=set()
for eng in lines.eng:
	for word in eng.split():
		all_eng_words.add(word)

# Vocabulary of Deutch 
all_deu_words=set()
for deu in lines.deu:
	for word in deu.split():
		all_deu_words.add(word)
			
print(str(len(all_eng_words)) + ' english words')
print(str(len(all_deu_words)) + ' deutch words')

source_word2idx = {}
source_idx2word = {}
target_word2idx = {}
target_idx2word = {}

count = 0
for word in all_eng_words:
	source_word2idx[word] = count
	count += 1

for word in source_word2idx:
	source_idx2word[source_word2idx[word]] = word

count = 0
for word in all_deu_words:
	target_word2idx[word] = count
	count += 1

for word in target_word2idx:
	target_idx2word[target_word2idx[word]] = word

BATCH_SIZE = 64
EMBED_SIZE = 256
UNITS = 1024

def gru(units):
	# If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
	# the code automatically does that.
		if tf.test.is_gpu_available():
			return tf.keras.layers.CuDNNGRU(units, 
				return_sequences=True, 
				return_state=True)
		else:
				return tf.keras.layers.GRU(units, 
					return_sequences=True, 
					return_state=True)

# no batch normalization => safe to use tf.keras.Model
class Encoder(tf.keras.Model):
	def __init__(self, enc_units):
		super(Encoder, self).__init__()
		self.embedding = tf.keras.layers.Embedding(len(source_word2idx),	EMBED_SIZE)
		self.gru = gru(enc_units)
		self.enc_units = enc_units
		
	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state=hidden)
		return output, state
	
	def init_hidden_state(self):
		return tf.zeros((BATCH_SIZE, self.enc_units))
		
class Decoder(tf.keras.Model):
	def __init__(self, dec_units):
		super(Decoder, self).__init__()
		self.embedding = tf.keras.layers.Embedding(len(target_word2idx), EMBED_SIZE)
		self.gru = gru(dec_units)
		self.fc = tf.keras.layers.Dense(len(target_word2idx))
		
		self.dec_units = dec_units
		
		# attention
		self.W1 = tf.keras.layers.Dense(dec_units)
		self.W2 = tf.keras.layers.Dense(dec_units)
		self.V = tf.keras.layers.Dense(1)
	
	def call(self, x, hidden, enc_output):
		# x : (1 word) : batch_size * 1
		# hidden size : batch_size X enc_units
		# enc_output : batch_size X n_words X enc_units
		hidden = tf.expand_dims(hidden, 1) # size : batch_size X 1 X enc_units
		
		score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden)))# size : batch_size X n_words X 1
		
		attention_weights = tf.nn.softmax(score, axis=1) # size : batch_size X n_words X 1
		
		context_vector = attention_weights * enc_output # size : batch_size X n_words X enc_units
		# sum all words : context_vector = e1*h1 + e2*h2 + ...
		context_vector = tf.reduce_sum(context_vector, axis=1) # size : batch_size X enc_units
		
		x = self.embedding(x) # size : batch_szie X 1 X EMBED_SIZE
		x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1) # size: batch_size X 1 X (enc_units + EMBED_SIZE)
		output, state = self.gru(x)
		output = tf.reshape(output, (-1, output.shape[2])) # output ~ state ??
		x = self.fc(output) # size : batch_size X vocab_size
		# state : batch_size X units
		return x, state, attention_weights
		
	def init_hidden_state(self):
		return tf.zeros((BATCH_SIZE, self.dec_units))
		
encoder = Encoder(UNITS)
decoder = Decoder(UNITS)
		
optimizer = tf.train.AdamOptimizer()
def loss_function(real, pred):
	# real: int32: batch_size X n_words
	# pred: float32: batch_size X n_words X vocab_size
	# batch_size = 1
	# => real: int32: n_words
	# => pred: float32: n_words X vocab_size
	mask = 1 - tf.equal(real, 0)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask # float32: batch_size X n_words
	loss = tf.reduce_sum(loss, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-6)
	return tf.reduce_mean(loss)

EPOCHS = 10
ids = np.arange(len(lines.eng))
X = tf.placeholder(tf.int32, [None, None])
HE = tf.placeholder(tf.float32, [None, UNITS])
OE, SE = encoder(X, HE)

HD = tf.placeholder(tf.float32, [None, UNITS])
W = tf.placeholder(tf.int32, [None, 1])
print(decoder(W, HD, SE))