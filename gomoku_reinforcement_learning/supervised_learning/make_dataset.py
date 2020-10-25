import numpy as np
import os

source_folder = './dataset/'
target_file = './data.txt'

board_width = 15
board_height = 15
fw = open(target_file, 'w')
sub_folders = os.listdir(source_folder)
count = 0
for sub_folder in sub_folders:
	files = os.listdir(source_folder + sub_folder)
	files = [file for file in files if file.endswith('psq')]
	for file in files:
		file_path = source_folder + sub_folder + '/' + file
		count+=1
		print(str(count) , file_path)
		f = open(file_path, 'r')
		s = f.readline()
		xs, ys = [], []
		while s!='' and s is not None:
			s = f.readline().strip()
			if s[0].isalpha():
				break
			y, x, _ = s.split(',')
			x = int(x)
			y = int(y)
			xs.append(x)
			ys.append(y)
		f.close()
		xs = np.int32(xs) - 1
		ys = np.int32(ys) - 1
		max_x = np.max(xs)
		min_x = np.min(xs)
		max_y = np.max(ys)
		min_y = np.min(ys)
		if max_x - min_x > board_width - 1 or max_y - min_y > board_height - 1:
			continue
		if max_x > board_width - 1:
			dx = max_x - board_width + 1
			xs = xs - dx
		if max_y > board_height - 1:
			dy = max_y - board_height + 1
			ys = ys - dy
		actions = list(ys * board_width + xs)
		fw.write(str(actions) + '\n')
fw.close()