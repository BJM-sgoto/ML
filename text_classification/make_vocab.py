import tensorflow as tf
import numpy as np
import os
import shutil

t = tf.keras.preprocessing.text.Tokenizer(
	num_words=1000,
	filters='!"#$%&()*+,-./:;<=>?@[\\]^_`´{|}~\'\t\n')
pos_source_folder = './data/train/pos/'
neg_source_folder = './data/train/neg/'

pos_target_folder = './data/train/gpos/'
neg_target_folder = './data/train/gneg/'

for file in os.listdir(pos_source_folder):
	print('Read file', pos_source_folder + file)
	f = open(pos_source_folder + file, 'r')
	try:
		t.fit_on_texts([f.read()])
		f.close()
		shutil.copy(pos_source_folder + file, pos_target_folder)
	except Exception as e:
		print('Error', e)
	finally:
		f.close()
		
for file in os.listdir(neg_source_folder):
	print('Read file', neg_source_folder + file)
	f = open(neg_source_folder + file, 'r')
	try:
		t.fit_on_texts([f.read()])
		f.close()
		shutil.copy(neg_source_folder + file, neg_target_folder)
	except Exception as e:
		print('Error', e)
	finally:
		f.close()
		
f = open('vocab.txt', 'w')
vocab = t.word_docs
vocab.pop('br')
vocab = sorted(vocab, key=vocab.__getitem__, reverse=True)
for item in vocab:
	f.write(item + ' ' + str(t.word_docs[item]) + '\n')
f.close()