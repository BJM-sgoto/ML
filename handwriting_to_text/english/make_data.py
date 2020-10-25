import os
import cv2
import numpy as np
import csv

word_file = './ascii/words.txt'
line_file = './ascii/lines.txt'
word_image_folder = './word_images/'

word_images = os.listdir(word_image_folder)

f_word = open(word_file, 'r')
f_line = open(line_file, 'r')
line_line = f_line.readline().strip()
line_word = f_word.readline().strip()
count_line_in_paragraph = 0

target_train_folder = './train_data/'
target_train_file = './train_text.txt'
target_test_folder = './test_data/'
target_test_file = './test_text.txt'
has_hyphen = False

test_rate = 0.01
train_file_content = ''
test_file_content = ''
count_to_exit = 0

dictionary = {}
target_dictionary_file = './dictionary.txt'

# remove files
for file in os.listdir(target_train_folder):
	os.remove(target_train_folder + file)
for file in os.listdir(target_test_folder):
	os.remove(target_test_folder + file)

while line_line:
	if not line_line.startswith('#'):
		sub_lines = line_line.split(' ')
		if sub_lines[1]=='ok': # line is ok
			line_id = sub_lines[0]
			sub_folders = line_id.split('-')
			sub_folder_1 = sub_folders[0]
			sub_folder_2 = sub_folders[0] + '-' + sub_folders[1]
			line_x = int(sub_lines[4])
			line_y = int(sub_lines[5])
			line_w = int(sub_lines[6])
			line_h = int(sub_lines[7])
			line_content = ' '.join(sub_lines[8:])
			line_content = line_content.lower()
			line_words = line_content.split('|')
			image_line = np.float32(np.ones([line_h, line_w, 3]) * 255)
			word_positions = []
			i=0
			for i in range(len(line_words)):
				line_word_id = sub_folder_2 + '-{:s}-{:02d}'.format(sub_folders[2], i)
				while line_word:
					if line_word.startswith(line_word_id):
						sub_words = line_word.split(' ')
						word_x = int(sub_words[3])
						word_y = int(sub_words[4])
						word_w = int(sub_words[5])
						word_h = int(sub_words[6])
						
						word_path = word_image_folder + sub_folder_1 + '/' + sub_folder_2 + '/' + line_word_id + '.png'
						image_word = np.float32(cv2.imread(word_path))
						word_x = word_x - line_x
						word_y = word_y - line_y
						
						word = sub_words[8]
						word_positions.append([word_x, word_w, word_y, word_h])
						image_line[word_y: word_y + word_h, word_x: word_x + word_w] = np.minimum(image_word, image_line[word_y: word_y + word_h, word_x: word_x + word_w])
						break
					line_word = f_word.readline().strip()
					
			
			if has_hyphen:
				line_words.pop(0)
				image_line = image_line[:, word_positions[0][1]:]
			
			if word.endswith('-'):
				# drop the last word
				image_line = image_line[:, : word_x]
				line_words.pop(-1)
				line = '|'.join(line_words)
				has_hyphen = True
			else:
				line = '|'.join(line_words)
				has_hyphen = False
			for dictionary_word in line_words:
				if dictionary_word in dictionary:
					dictionary[dictionary_word]+=1
				else:
					dictionary[dictionary_word] = 1
			
			if np.random.uniform()<test_rate:
				line_path = target_test_folder + line_id + '.png'
				cv2.imwrite(line_path, image_line)
				print(line_path)
				test_file_content+= line_id + '.png\t' + line + '\t' + str(word_positions) + '\n'
			else:
				line_path = target_train_folder + line_id + '.png'
				cv2.imwrite(line_path, image_line)
				print(line_path)
				train_file_content+= line_id + '.png\t' + line + '\t' + str(word_positions) + '\n'
			#test
			#count_to_exit+=1
			#if count_to_exit>=200:
			#	break
		else:
			has_hyphen = False# reset variable
	line_line = f_line.readline().strip()
	
	
f_word.close()
f_line.close()

f_train = open(target_train_file, 'w')
f_train.write(train_file_content)
f_train.close()

f_test = open(target_test_file, 'w')
f_test.write(test_file_content)
f_test.close()

f_dictionary = open(target_dictionary_file, 'w')
dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: -item[1])}
for dictionary_word in dictionary:
	f_dictionary.write(dictionary_word + '\t' + str(dictionary[dictionary_word]) + '\n')
f_dictionary.close()