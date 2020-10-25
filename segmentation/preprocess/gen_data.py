import cv2
import numpy as np
import os

source_image_folder = './ori/'
source_mark_folder = './mark_polygon/'
target_image_folder = './train_image/'
target_mark_folder = './train_mark/'

num_rotation = 10
count = 0
rate = 2/3
cut_size = int(384*rate)

mark_files = os.listdir(source_mark_folder)
for mark_file in mark_files:
	print(mark_file)
	f = open(source_mark_folder + mark_file, 'r')
	s = f.readline()
	polygons = []
	while s!='':
		numbers = s.strip().split(' ')
		polygon = []
		for i in range(0,len(numbers),2):
			polygon.append([float(numbers[i]), float(numbers[i+1])])
		polygon = np.float32(polygon)
		polygons.append(polygon)
		s=f.readline()
	f.close()


	for polygon in polygons:
		polygon = polygon*rate
		img = cv2.imread(source_image_folder + mark_file[:-4])
		h, w, _ = img.shape
		w, h = int(w*rate), int(h*rate)
		img = cv2.resize(img, (w, h))
		min_xy = np.min(polygon, axis=0)
		max_xy = np.max(polygon, axis=0)
		center_xy = (min_xy + max_xy)/2
		start_x = int(center_xy[0] - cut_size)
		if start_x<0:
			start_x = 0
		start_y = int(center_xy[1] - cut_size)
		if start_y<0:
			start_y = 0
		end_x = start_x + 2*cut_size
		end_y = start_y + 2*cut_size
		if end_x>w:
			end_x = w
			start_x = end_x - 2*cut_size
		if end_y>h:
			end_y = h
			start_y = end_y - 2*cut_size
		sub_img = img[start_y:end_y, start_x:end_x]
		center_xy[0] -= start_x
		center_xy[1] -= start_y
		polygon[:, 0] -= start_x
		polygon[:, 1] -= start_y
		
		# translate images
		M = np.float32([[1,0,cut_size - center_xy[0]],[0,1, cut_size-center_xy[1]]])
		sub_img = cv2.warpAffine(sub_img, M, (cut_size*2, cut_size*2))
		polygon[:, 0] += cut_size - center_xy[0]
		polygon[:, 1] += cut_size - center_xy[1]
		center_xy[0] = cut_size
		center_xy[1] = cut_size
		
		# rotate images
		for i in range(num_rotation):
			random_number = np.random.uniform()
			print('random numbers', random_number)
			angle = random_number*360
			M = cv2.getRotationMatrix2D((cut_size, cut_size),angle,1)
			r_img = cv2.warpAffine(sub_img, M, (cut_size*2, cut_size*2))
						
			r_polygon = np.copy(polygon)
			r_polygon[:,0] -= center_xy[0]
			r_polygon[:,1] -= center_xy[1]
			
			new_x = r_polygon[:,0] * np.cos(-random_number*2*np.pi) - r_polygon[:,1] * np.sin(-random_number*2*np.pi)
			new_y = r_polygon[:,0] * np.sin(-random_number*2*np.pi) + r_polygon[:,1] * np.cos(-random_number*2*np.pi)
			
			r_polygon[:,0] = new_x + cut_size/2
			r_polygon[:,1] = new_y + cut_size/2
			
			min_xy = np.min(r_polygon, axis=0)
			max_xy = np.max(r_polygon, axis=0)
			
			obj_w = np.max(new_x) - np.min(new_x)
			obj_h = np.max(new_y) - np.min(new_y)
			space_w = cut_size - obj_w
			space_h = cut_size - obj_h
			
			offset_w = (np.random.uniform()-0.5)* space_w
			offset_h = (np.random.uniform()-0.5)* space_h
			
			start_x = int(center_xy[0] - cut_size/2 + offset_w)
			start_y = int(center_xy[1] - cut_size/2 + offset_h)
			end_x = int(start_x + cut_size)
			end_y = int(start_y + cut_size)
			
			r_img = r_img[start_y:end_y, start_x:end_x]
			cv2.imwrite(target_image_folder + '{:06d}.jpg'.format(count), r_img)
			
			r_polygon[:,0] -= offset_w
			r_polygon[:,1] -= offset_h
			r_polygon = np.int32(r_polygon)
			r_polygon = np.int32([r_polygon])
			
			mark_img = np.zeros([cut_size, cut_size])
			cv2.fillPoly(mark_img, r_polygon, 255)
			cv2.imwrite(target_mark_folder + '{:06d}.jpg'.format(count), mark_img)
			print(target_mark_folder + '{:06d}.jpg'.format(count))
			
			count+=1