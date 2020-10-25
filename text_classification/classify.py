import tensorflow as tf
import numpy as np


train_file = './data/train/train_data.txt'
batch_size = 200
resume = False

train_data = []
f = open(train_file, 'r')
s = f.readline()
dataset = {'input': [], 'output': []}
while s:
	line = s.strip()
	data_input = list(np.fromstring(line[1:-3], dtype=int, sep=' '))
	dataset['input'].append(data_input)
	data_output = np.float32(line[-1])
	dataset['output'].append(data_output)
	s = f.readline()	
f.close()

# training
if not resume:
	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(10000 + 1, 64),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')])

	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])
else:
	model = tf.keras.models.load_model('model.h5')

model.summary()

'''
num_data = len(dataset['input'])
ids = np.arange(num_data)

for i in range(10):
	np.random.shuffle(ids)
	for j in range(0, num_data, batch_size):
		end_j = min(num_data, j+batch_size)
		
		# make input, output and 
		data_input = []
		data_output = []
		max_len_input = 0
		for k in range(j, end_j):
			data_input_item = dataset['input'][ids[k]]
			data_input.append(data_input_item)
			if len(data_input_item)>max_len_input:
				max_len_input = len(data_input_item)
			data_output.append(dataset['output'][ids[k]])
		
		print('Max length of input', max_len_input)
		
		# padding 0
		for data_input_item in data_input:
			data_input_item += [0]*(max_len_input-len(data_input_item))
		data_input = np.int32(data_input)	
		data_output = np.float32(data_output)
		
		model.fit(x=data_input, y=data_output, batch_size=end_j - j)
	
model.save('model.h5')
'''

'''
# test model 
model = tf.keras.models.load_model('./model.h5')
vocab_max_size = 10000

vocab = {}
vocab_file = './clean_vocab_ubuntu.txt'
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
'''