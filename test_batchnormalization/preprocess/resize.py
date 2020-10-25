import os
import numpy as np
import cv2

source_folder = './data/'
target_train_dog_folder = './train_dog/'
target_train_cat_folder = './train_cat/'
target_test_dog_folder = './test_dog/'
target_test_cat_folder = './test_cat/'

cat_images = [x for x in os.listdir(source_folder) if x.startswith('cat')]
dog_images = [x for x in os.listdir(source_folder) if x.startswith('dog')]
for cat_image in cat_images:
	img = cv2.imread(source_folder + cat_image)
	img = cv2.resize(img, (276,276))
	target_cat_folder = target_train_cat_folder
	if np.random.rand()>0.9:
		target_cat_folder = target_test_cat_folder
	cv2.imwrite(target_cat_folder + cat_image, img)
	print(cat_image)
	
for dog_image in dog_images:
	img = cv2.imread(source_folder + dog_image)
	img = cv2.resize(img, (276,276))
	target_dog_folder = target_train_dog_folder
	if np.random.rand()>0.9:
		target_dog_folder = target_test_dog_folder
	cv2.imwrite(target_dog_folder + dog_image, img)
	print(dog_image)