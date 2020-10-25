import cv2
import os
import numpy as np

source_folder = './source/'
target_image_folder = './image/'
target_edge_folder = './edge/'
count = 0
for sub_folder1 in os.listdir(source_folder):
	sub_path1 = source_folder + sub_folder1 + '/'
	for sub_folder2 in os.listdir(sub_path1):
		sub_path2 = sub_path1 + sub_folder2 + '/'
		for sub_folder3 in os.listdir(sub_path2):
			sub_path3  = sub_path2 + sub_folder3 + '/'
			for image_name in os.listdir(sub_path3):
				image = cv2.imread(sub_path3 + image_name)
				image = cv2.resize(image, (128, 128))
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				edge = cv2.Canny(gray,100,200)
				cv2.imwrite(target_image_folder + '{:06d}.jpg'.format(count), image)
				cv2.imwrite(target_edge_folder + '{:06d}.jpg'.format(count), edge)
				print(count)
				count += 1