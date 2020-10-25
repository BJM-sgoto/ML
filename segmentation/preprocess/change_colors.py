import cv2
import os
import numpy as np

source_folder = './ori/'
target_folder = './image/'

files = os.listdir(source_folder)
for file in files:
	img = cv2.imread(source_folder + file)
	w, h, _ = img.shape
	w = int(w/2)
	h = int(h/2)
	img = cv2.resize(img,(h, w))
	for i in range(w):
		for j in range(h):
			if np.sum(np.equal(img[i,j], [255,0,255]))==3:
				img[i,j,0] = 254
	cv2.imwrite(target_folder + file, img)
	print(file)