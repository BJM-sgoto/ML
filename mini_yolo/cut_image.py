import os
import cv2
import numpy as np

source_image_folder = './full_dataset/train_image/'
source_label_folder = './full_dataset/train_label/'
target_object_folder = './full_dataset/cut_image/'
target_nonobject_folder = './non_object/'
background_folder = './background/'

backgrounds = [cv2.imread(background_folder + x) for x in os.listdir(background_folder)]

def rotate_image(img, rectangle):
	background = backgrounds[np.random.randint(low=0, high=len(backgrounds))]
	h, w, _ = background.shape
	start_cut_x = np.random.randint(low=0, high=w-640+1)
	end_cut_x = start_cut_x + 640
	start_cut_y = np.random.randint(low=0, high=h-480+1)
	end_cut_y = start_cut_y + 480
	background = background[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
	background = background.copy()
	
	rotate_angle = np.random.randint(low=0, high=360)
	M = cv2.getRotationMatrix2D((rectangle[0], rectangle[1]),rotate_angle,1)
	img = img + 1
	img = cv2.warpAffine(img, M, (640, 480))
	mask = np.where(img>0.5, 1, 0)
	img = (img - 1) * mask + background * (1 - mask)
	angle = rectangle[4] - rotate_angle / 180 * np.pi
	if angle < 0:
		angle += 2*np.pi
	return img, angle
	
def compute_overlap(grid, rectangle):
	center_x = rectangle[0]
	center_y = rectangle[1]
	rectangle_width = rectangle[2]
	rectangle_height = rectangle[3]
	angle = rectangle[4]
	
	img = np.zeros([480, 640])
	half_width = rectangle_width / 2
	half_height = rectangle_height / 2
	sin_angle = np.sin(angle)
	cos_angle = np.cos(angle)
	x1 = center_x - half_width * cos_angle + half_height * sin_angle
	y1 = center_y - half_width * sin_angle - half_height * cos_angle
	x2 = center_x + half_width * cos_angle + half_height * sin_angle
	y2 = center_y + half_width * sin_angle - half_height * cos_angle
	x3 = center_x + half_width * cos_angle - half_height * sin_angle
	y3 = center_y + half_width * sin_angle + half_height * cos_angle
	x4 = center_x - half_width * cos_angle - half_height * sin_angle
	y4 = center_y - half_width * sin_angle + half_height * cos_angle
	
	pnts = np.int32(np.round([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]))
	img = cv2.fillPoly(img, pnts, 255)
	n_vcells = len(grid)
	n_hcells = len(grid[0])
	ret = np.float32(np.zeros([n_vcells, n_hcells]))
	area = np.sum(img)
	for i in range(n_vcells):
		for j in range(n_hcells):
			ret[i,j] = np.sum(img[grid[i,j,1]: grid[i,j,3], grid[i,j,0]: grid[i,j,2]])/area
	return ret

grid_width = 640
grid_height = 480
cell_width = 142
cell_height = 142
jump_x = 16
jump_y = 16
start_center_x = 8
start_center_y = 8
n_hcells = int((grid_width - start_center_x)/ jump_y) + 1
n_vcells = int((grid_height - start_center_y)/ jump_x) + 1
grid = np.int32(np.zeros([n_vcells, n_hcells, 4]))
for i in range(n_vcells):
	for j in range(n_hcells):
		grid[i,j,0] = max(int(start_center_x - cell_width/2 + jump_x*j), 0)
		grid[i,j,1] = max(int(start_center_y - cell_height/2 + jump_y*i), 0)
		grid[i,j,2] = min(int(grid[i,j,0] + cell_width), grid_width)
		grid[i,j,3] = min(int(grid[i,j,1] + cell_height), grid_height)

label_files = os.listdir(source_label_folder)
count = 0
for label_file in label_files:
	image_file = label_file[:-4]
	f = open(source_label_folder + label_file, 'r')
	s = f.read().strip().split()
	f.close()
	rectangle = [float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4])]
	img = cv2.imread(source_image_folder + image_file)
	for i in range(50):
		new_img, angle = rotate_image(img, rectangle)
		new_rectangle = np.copy(rectangle)
		new_rectangle[4] = angle
		overlap = compute_overlap(grid, new_rectangle)
		object_pos = np.where(overlap>0.9)
		non_object_pos = np.where(overlap<=0.5)
		ids1 = np.arange(len(object_pos[0]))
		ids2 = np.arange(len(non_object_pos[0]))
		np.random.shuffle(ids1)
		np.random.shuffle(ids2)
		max_j = min(len(ids1), len(ids2))
		max_j = min(15, max_j)
		for j in range(1):
			# object image
			start_cut_y = max(0, object_pos[0][ids1[j]]*16 - 71)
			start_cut_x = max(0, object_pos[1][ids1[j]]*16 - 71)
			end_cut_x = start_cut_x + 142
			if end_cut_x > 640:
				end_cut_x = 640
				start_cut_x = 640 - 142
			end_cut_y = start_cut_y + 142
			if end_cut_y > 480:
				end_cut_y = 480
				start_cut_y = 480 - 142
			img_object = new_img[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
			cv2.imwrite(target_object_folder + '{:06d}.jpg'.format(count), img_object)
			f = open('./full_dataset/cut_label/' + '{:06d}.jpg.txt'.format(count), 'w')
			f.write(str(rectangle[0] - start_cut_x) + ' ' + str(rectangle[1] - start_cut_y) + ' ' + str(rectangle[2]) + ' ' + str(rectangle[3]) + ' ' + str(angle))
			f.close()
			print(target_object_folder + '{:06d}.jpg'.format(count))
			
			'''
			# non object image
			start_cut_y = max(0, non_object_pos[0][ids2[j]]*16 - 71)
			start_cut_x = max(0, non_object_pos[1][ids2[j]]*16 - 71)
			end_cut_x = start_cut_x + 142
			if end_cut_x > 640:
				end_cut_x = 640
				start_cut_x = 640 - 142
			end_cut_y = start_cut_y + 142
			if end_cut_y > 480:
				end_cut_y = 480
				start_cut_y = 480 - 142
			img_object = new_img[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
			cv2.imwrite(target_nonobject_folder + '{:06d}.jpg'.format(count), img_object)
			print(target_nonobject_folder + '{:06d}.jpg'.format(count))
			'''
			count = count + 1