import os
import cv2

source_folder = './train_zebra/'
target_folder = './small_zebra/'
count = 0
for file in os.listdir(source_folder):
	image = cv2.imread(source_folder + file)
	image = cv2.resize(image, (148, 148))
	cv2.imwrite(target_folder + '{:06d}.jpg'.format(count), image)
	count += 1
	print(count)