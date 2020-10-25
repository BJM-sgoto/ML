import numpy as np 
import cv2
import os

np.random.seed(1)

NUM_SAMPLES = 40000

TRAIN_TARGET_IMAGE_FOLDER = './train_dataset/'
TEST_TARGET_IMAGE_FOLDER = './test_dataset/'

TRAIN_TARGET_OUTPUT_FILE = './train_img_text.txt'
TEST_TARGET_OUTPUT_FILE = './test_img_text.txt'

TEST_RATE = 0.02

LOW_CHARS = ['j', 'p', 'q', 'g', 'y']

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

CHARS = []
for i in range(ord('a'), ord('z') + 1):
	CHARS.append(chr(i))

train_output = ''
test_output = ''

# remove all files in folders

for file in os.listdir(TRAIN_TARGET_IMAGE_FOLDER):
	os.remove(TRAIN_TARGET_IMAGE_FOLDER + file)
for file in os.listdir(TEST_TARGET_IMAGE_FOLDER):
	os.remove(TEST_TARGET_IMAGE_FOLDER + file)

max_image_width = 0
for i in range(NUM_SAMPLES):
	font = np.random.choice(FONTS)
	text = np.random.choice(CHARS)
	# make image
	#background_color = np.random.uniform(low=0, high=255.0, size=[3])
	background_color = np.zeros([3])
	#text_color = np.random.uniform(low=0, high=255.0, size=[3])
	text_color = np.ones([3]) * 255
	font_scale = 1.0
	text_thickness = 1
	text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
	while text_size[0]<46+3*16:
		text+=np.random.choice(CHARS)
		text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
	text = text[:-1]
	text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
	print(i, text, text_size, baseline, 'Font:', font)
	image_width, image_height = text_size
	max_image_width = max(max_image_width, image_width)
	
	for low_char in LOW_CHARS:
		if low_char in text:
			image_height+=baseline
			break
	else:
		baseline = 0
	image = np.ones([image_height, image_width, 3], dtype=np.float32) * background_color
	text_left = 0
	text_bottom = image_height - 1 - baseline
	cv2.putText(image, text, (text_left, text_bottom), font, font_scale, text_color, text_thickness)
	image_file = '{:05d}.png'.format(i)
	accum_text_lens = []
	for j in range(1, len(text)+1):
		sub_text = text[:j]
		(sub_text_len, _), _ = cv2.getTextSize(sub_text, font, font_scale, text_thickness)
		accum_text_lens.append(sub_text_len)
	if np.random.rand()>TEST_RATE:
		output_image_path = TRAIN_TARGET_IMAGE_FOLDER + image_file
		cv2.imwrite(output_image_path, image)
		train_output += image_file + '\t' + text + '\t' + str(accum_text_lens) + '\n'
	else:
		output_image_path = TEST_TARGET_IMAGE_FOLDER + image_file
		cv2.imwrite(output_image_path, image)
		test_output += image_file + '\t' + text + '\t' + str(accum_text_lens) + '\n'

f = open(TRAIN_TARGET_OUTPUT_FILE, 'w')
f.write(train_output)
f.close()
f = open(TEST_TARGET_OUTPUT_FILE, 'w')
f.write(test_output)
f.close()
print(max_image_width)