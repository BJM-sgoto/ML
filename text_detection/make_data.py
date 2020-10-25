import numpy as np
import cv2
import os

dictionary = {}
N_SAMPLES = 10000
FONTS = [
cv2.FONT_HERSHEY_SIMPLEX, # ok
#cv2.FONT_HERSHEY_COMPLEX_SMALL, # this font is too small
cv2.FONT_HERSHEY_DUPLEX, # ok
#cv2.FONT_HERSHEY_PLAIN, # this font is too small
#cv2.FONT_HERSHEY_SCRIPT_COMPLEX, # curly
#cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, # curly
cv2.FONT_HERSHEY_SIMPLEX, # ok
cv2.FONT_HERSHEY_TRIPLEX, # ok
cv2.FONT_ITALIC # ok
]
LOW_CHARS = ['j', 'p', 'q', 'g', 'y']
IMG_WIDTH = 512
IMG_HEIGHT = 256
CHARS = []
TEST_RATE = 0.05
TARGET_TRAIN_FOLDER = './train_dataset/'
TARGET_TEST_FOLDER = './test_dataset/'

for i in range(ord('a'), ord('z') + 1):
	CHARS.append(chr(i))
	
for file in os.listdir(TARGET_TRAIN_FOLDER):
	os.remove(TARGET_TRAIN_FOLDER + file)
for file in os.listdir(TARGET_TEST_FOLDER):
	os.remove(TARGET_TEST_FOLDER + file)	
	
min_height = 1000
max_height = 0

for i in range(N_SAMPLES):
	top = 0
	# init image
	img = np.zeros([IMG_HEIGHT, IMG_WIDTH], np.float32)
	# init file content
	file_content = ''
	while True:
		# add space
		top += np.random.randint(low=5, high=10)
		
		# set font, font scale ...
		font = np.random.choice(FONTS)
		font_scale = np.random.uniform(low=0.8, high=1.5)
		text_thickness = 1
		max_len_text = np.random.uniform(low=IMG_WIDTH/2, high=IMG_WIDTH)
		# make text
		text = np.random.choice(CHARS)
		(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
		text_ends = [text_width]
		while text_width<max_len_text:
			text+=np.random.choice(CHARS)
			(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
			text_ends.append(text_width)
		text = text[:-1]
		text_ends.pop(-1)
		(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
		
		if top + text_height >= IMG_HEIGHT:
			break
		min_height = min(min_height, text_height)
		max_height = max(max_height, text_height)
		# draw new line
		pad_left = np.random.randint(low=0, high=IMG_WIDTH-text_width)
		text_ends = [text_end + pad_left for text_end in text_ends]
		cv2.putText(img, text, (pad_left, top + text_height), font, font_scale, 255, text_thickness)
		top = top + text_height + baseline
		# make file content
		file_content += text + '\t' + str([pad_left, top - text_height - baseline, text_width, text_height + baseline]) + '\t' + str(text_ends) + '\n'
	target_folder = TARGET_TRAIN_FOLDER
	if np.random.uniform()<TEST_RATE:
		target_folder = TARGET_TEST_FOLDER
	
	image_name = '{:06d}.png'.format(i)
	text_name = '{:06d}.txt'.format(i)
	cv2.imwrite(target_folder + image_name, img)
	f = open(target_folder + text_name, 'w')
	f.write(file_content)
	f.close()
	print(target_folder + '->' + image_name + ': ' + text_name)
print('Min height', min_height)
print('Max height', max_height)