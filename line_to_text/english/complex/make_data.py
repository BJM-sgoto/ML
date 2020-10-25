import numpy as np 
import cv2

np.random.seed(1)

TEXT_LEN = 100
NUM_SAMPLES = 10000

TRAIN_TARGET_IMAGE_FOLDER = './train_dataset/'
TEST_TARGET_IMAGE_FOLDER = './test_dataset/'
TRAIN_TARGET_OUTPUT_FILE = './train_img_text.txt'
TEST_TARGET_OUTPUT_FILE = './test_img_text.txt'

LOW_CHARS = ['p', 'q', 'g', 'y']

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, 
#cv2.FONT_HERSHEY_COMPLEX_SMALL, # this font is too small
cv2.FONT_HERSHEY_DUPLEX,
#cv2.FONT_HERSHEY_PLAIN, # this font is too small
cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
cv2.FONT_HERSHEY_SIMPLEX,
cv2.FONT_HERSHEY_TRIPLEX,
cv2.FONT_ITALIC]

CHARS = []
for i in range(ord('a'), ord('z') + 1):
	CHARS.append(chr(i))

train_output = ''
test_output = ''
for i in range(NUM_SAMPLES):
	font = np.random.choice(FONTS)
	char_ids = list(np.random.randint(low=0, high=len(CHARS), size=[TEXT_LEN]))
	text = [CHARS[i] for i in char_ids]
	text = ''.join(text)
	# make image
	background_color = np.random.uniform(low=0, high=255.0, size=[3])
	text_color = np.random.uniform(low=0, high=255.0, size=[3])
	font_scale = 1.0
	text_thickness = 1
	text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
	print(i, text, text_size, baseline, 'Font:', font)
	image_width, image_height = text_size
	image_height+=2
	for low_char in LOW_CHARS:
		if low_char in text:
			image_height+=baseline
			break
	else:
		baseline = 0
	image = np.ones([image_height, image_width, 3], dtype=np.float32) * background_color
	text_left = 0
	text_bottom = image_height - 3 - baseline
	cv2.putText(image, text, (text_left, text_bottom), font, font_scale, text_color, text_thickness)
	image_file = '{:05d}.png'.format(i)
	accum_text_lens = []
	for j in range(1, TEXT_LEN+1):
		sub_text = text[:j]
		(sub_text_len, _), _ = cv2.getTextSize(sub_text, font, font_scale, text_thickness)
		accum_text_lens.append(sub_text_len)
	if np.random.rand()>0.02:
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