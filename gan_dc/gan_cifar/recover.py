import cv2
import pickle
import numpy as np
source_file = 'data_batch_'
target_folder = './train/'

for k in range(1,6):
	f = open(source_file + str(k), 'rb')
	data = pickle.load(f, encoding='bytes')
	num_data = len(data[b'labels'])
	for i in range(num_data):
		image = data[b'data'][i]
		image = np.reshape(image, [3 ,32, 32])
		image = np.flip(image, axis=0)
		image = np.transpose(image, [1,2,0])
		label = data[b'labels'][i]
		filename = str(data[b'filenames'][i], 'utf-8')
		cv2.imwrite(target_folder + str(label) + '/' + filename, image)
		print(filename)
	f.close()