# reference: https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=en

import tensorflow as tf

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

NUM_EXAMPLES = 30000

class Processor:
	def unicode_to_ascii(self, s):
		return ''.join(c for c in unicodedata.normalize('NFD', s)
			if unicodedata.category(c) != 'Mn')

	def preprocess_sentence(self, w):
		w = self.unicode_to_ascii(w.lower().strip())
		w = re.sub(r"([?.!,¿])", r" \1 ", w)
		w = re.sub(r'[" "]+', " ", w)
		w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
		w = w.rstrip().strip()
		w = '<start> ' + w + ' <end>'
		return w
		
	def create_dataset(self, path, num_examples):
		lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
		word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
		return zip(*word_pairs)
	
	def max_length(self, tensor):
		return max(len(t) for t in tensor)
	
	def tokenize(self, lang):
		lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
		lang_tokenizer.fit_on_texts(lang)
		tensor = lang_tokenizer.texts_to_sequences(lang)
		tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
		return tensor, lang_tokenizer
	
	def load_dataset(self, path, num_examples=None):
		target_lang, source_lang = self.create_dataset(path, num_examples)
		source_tensor, source_lang_tokenizer = self.tokenize(source_lang)
		target_tensor, target_lang_tokenizer = self.tokenize(target_lang)
		return source_tensor, target_tensor, source_lang_tokenizer, target_lang_tokenizer

class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
		super(Encoder, self).__init__()
		self.batch_size = batch_size
		self.enc_units = enc_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(
			self.enc_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer='glorot_uniform')
	
	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state=hidden)
		return output, state
	
	def initialize_hidden_state(self):
		return tf.zeros((self.batch_size, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)
	
	def call(self, query, values):
		hidden_with_time_axis = tf.expand_dims(query, axis=1)
		score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
		attention_weights = tf.nn.softmax(score, axis=1)
		context_vector = attention_weights * values
		context_vector = tf.reduce_sum(context_vector, axis=1)
		return context_vector, attention_weights
	
class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
		super(Decoder, self).__init__()
		self.batch_size = batch_size
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(
			self.dec_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer='glorot_uniform')
		self.fc = tf.keras.layers.Dense(vocab_size)
		self.attention = BahdanauAttention(self.dec_units)
	
	def call(self, x, hidden, enc_output):
		context_vector, attention_weights = self.attention(hidden, enc_output)
		x = self.embedding(x)
		x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)
		output, state = self.gru(x)
		output = tf.reshape(output, (-1, output.shape[2]))
		x = self.fc(output)
		return x, state, attention_weights
	
	
	
	
processor = Processor()
source_tensor, target_tensor, source_lang_tokenizer, target_lang_tokenizer = processor.load_dataset('./spa.txt', NUM_EXAMPLES)
source_tensor_train, source_tensor_val, target_tensor_train, target_tensor_val = train_test_split(source_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(source_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(source_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_source_size = len(source_lang_tokenizer.word_index)+1 # ?
vocab_target_size = len(target_lang_tokenizer.word_index)+1 # ?
dataset = tf.data.Dataset.from_tensor_slices((source_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
#example_source_batch, example_target_batch = next(iter(dataset))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss = loss_object(real, pred)
	mask = tf.cast(mask, dtype = loss.dtype)
	loss = loss * mask
	return tf.reduce_mean(loss)

encoder = Encoder(vocab_source_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
	encoder=encoder,
	decoder=decoder)

@tf.function
def train_step(source, target, enc_hidden):
	loss = 0.0
	with tf.GradientTape() as tape:
		enc_output, enc_hidden = encoder(source, enc_hidden)
		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<start>']]*BATCH_SIZE, axis=1)
		for t in range(1, target.shape[1]):
			predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
			loss += loss_function(target[:, t], predictions)
			dec_input = tf.expand_dims(target[:, t], axis=1)
	batch_loss = loss / int(target.shape[1])
	variables = encoder.trainable_variables + decoder.trainable_variables
	gradients = tape.gradient(loss, variables)
	optimizer.apply_gradients(zip(gradients, variables))
	return batch_loss

EPOCHS = 10
enc_hidden = encoder.initialize_hidden_state()
for epoch in range(EPOCHS):
	start = time.time()
	for (batch, (source, target)) in enumerate(dataset.take(steps_per_epoch)):
		batch_loss = train_step(source, target, enc_hidden)
		print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
	checkpoint.save(file_prefix = checkpoint_prefix)

	