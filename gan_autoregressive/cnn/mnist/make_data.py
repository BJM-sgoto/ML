import keras
import os
from keras.datasets import mnist
from PIL import Image
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_test
y = y_test

n = x.shape[0]
source_folder = './train/'
for i in range(10):
	path = source_folder + str(i)
	if not os.path.exists(path):
		os.mkdir(path)
count = 60000
for i in range(n):
	image = Image.fromarray(x[i])
	image.save(source_folder + str(y[i]) + '/{:06d}.png'.format(count))
	print(count)
	count+=1