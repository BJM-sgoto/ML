import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`´{|}~\'\t\n'
pos_source_folder = './data/train/gpos/'
neg_source_folder = './data/train/gneg/'
train_file = './data/train/train_data.txt'

vocab_max_size = 10000

vocab = {}
vocab_file = './clean_vocab.txt'
f = open(vocab_file, 'r')
s = f.readline()
count = 0

while s:
	items = s.split(' ')
	vocab[items[0]] = count
	count += 1
	s = f.readline()
	if count>=vocab_max_size:
		break
f.close()

# add positive data
pos_train_data = []
for file in os.listdir(pos_source_folder):
	f = open(pos_source_folder + file, 'r')
	print('Reading', pos_source_folder + file)
	text = f.read()
	words = keras.preprocessing.text.text_to_word_sequence(text, filters=filters, lower=True, split=' ')
	para = []
	for word in words:
		para.append(vocab.get(word, -1) + 1)
	pos_train_data.append(para)
	f.close()

# add negative data
neg_train_data = []
for file in os.listdir(neg_source_folder):
	f = open(neg_source_folder + file, 'r')
	print('Reading', neg_source_folder + file)
	text = f.read()
	words = keras.preprocessing.text.text_to_word_sequence(text, filters=filters, lower=True, split=' ')
	para = []
	for word in words:
		para.append(vocab.get(word, -1) + 1)
	neg_train_data.append(para)
	f.close()

f = open(train_file, 'w')
for train_item in pos_train_data:
	f.write(np.array2string(np.int32(train_item), max_line_width=10000) + ' 1\n')
for train_item in neg_train_data:
	f.write(np.array2string(np.int32(train_item), max_line_width=10000) + ' 0\n')
f.close()
