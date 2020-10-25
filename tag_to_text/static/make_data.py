import numpy as np
import cv2
import random

np.random.seed(1)

dict_file = './dict.txt'
train_target_image_folder = './train_dataset/'
test_target_image_folder = './test_dataset/'
train_target_output_file = './train_img_text.txt'
test_target_output_file = './test_img_text.txt'
low_chars = ['p', 'q', 'g', 'y']

fonts = [cv2.FONT_HERSHEY_SIMPLEX, 
cv2.FONT_HERSHEY_COMPLEX_SMALL,
cv2.FONT_HERSHEY_DUPLEX,
#cv2.FONT_HERSHEY_PLAIN, # this font is too small
cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
cv2.FONT_HERSHEY_SIMPLEX,
cv2.FONT_HERSHEY_TRIPLEX,
cv2.FONT_ITALIC]

f = open(dict_file, 'r')
word_dict = eval(f.read())
word_dict = [word for word in word_dict if word.isalpha()]
dict_size = len(word_dict)
f.close()
train_output = ''
test_output = ''
max_img_height = 39 # already knew
for i in range(10000):
	text_len = 4  # TODO change back to 2
	word_ids = list(np.random.randint(low=0, high=dict_size, size=[text_len]))
	text = [word_dict[i] for i in word_ids]
	text = ''.join(text)
	# TODO remove this part later
	text = list(text)
	random.shuffle(text)
	text = ''.join(text)
	text = text[:text_len]
	
	#background_color = np.random.uniform(low=0, high=255.0, size=[3]) #TODO
	background_color = np.zeros([3], dtype=np.float32)
	
	font_scale = 1.0
	text_thickness = np.random.choice([1,2])
	#text_color = np.float32(np.random.uniform(low=0, high=255.0, size=[3])) #TODO
	text_color = np.ones([3], dtype=np.float64) * 255
	font = np.random.choice(fonts)
	text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
	text_width, text_height = text_size
	img_height = text_height + 2 * text_thickness
	for low_char in low_chars:
		if low_char in text:
			img_height += baseline
			break
	else:
		baseline = 0
	img_width = text_width
	text_left = 0
	text_bottom = text_height + text_thickness
	
	padded_img_width = int((text_width - 46)//32 + 1) * 32 + 46
	padded_img_height = 46
	text_left = np.random.randint(low = 0, high = padded_img_width - img_width)
	text_bottom = np.random.randint(low = img_height - baseline, high = padded_img_height - baseline)
	img = np.ones([padded_img_height, padded_img_width, 3]) * background_color
	cv2.putText(img, text, (text_left, text_bottom), font, font_scale, text_color, text_thickness)
	print('ID:', i, ',Text Size:', (text_width, text_height), ',Image Size', (padded_img_width, padded_img_height), ' Text pos:', (text_left, text_bottom), ',Font:', font, ',Thickness:', text_thickness, ', Baseline', baseline)
	image_file = '{:05d}.jpg'.format(i)
	if np.random.rand()>0.01:
		output_image_path = train_target_image_folder + image_file
		cv2.imwrite(output_image_path, img)
		train_output += image_file + '\t' + text + '\n'
	else:
		output_image_path = test_target_image_folder + image_file
		cv2.imwrite(output_image_path, img)
		test_output += image_file + '\t' + text + '\n'
	
f = open(train_target_output_file, 'w')
f.write(train_output)
f.close()
f = open(test_target_output_file, 'w')
f.write(test_output)
f.close()
print(max_img_height)