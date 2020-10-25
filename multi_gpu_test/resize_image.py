import cv2
import os
import numpy as np

source_folder = './train/cat/'
target_folder = './small_train/cat/'
for file in os.listdir(source_folder):
	img = np.float32(cv2.imread(source_folder + file))
	img = cv2.resize(img, (244,244))
	cv2.imwrite(target_folder + file, img)
	print(file)