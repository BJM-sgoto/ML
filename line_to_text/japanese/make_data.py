import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import os

N_SAMPLES = 40000
#CUT_WIDTH = 46+16*5
CUT_WIDTH = 36+8*11
train_folder = './train_dataset/'
test_folder = './test_dataset/'
img = np.zeros((200,400,3),np.uint8)
b,g,r,a = 255,255,255,0
#fontpaths = ["simsun.ttc","UDDigiKyokashoN-B.ttc","UDDigiKyokashoN-R.ttc","msmincho.ttc","HGRME.TTC","HGRMB.TTC","HGRPRE.TTC","HGRGY.TTC","HGRKK.TTC"]
fontpaths = ["meiryob.ttc", "meiryo.ttc", "msgothic.ttc","msmincho.ttc","mingliu.ttc", "simsun.ttc"]
kanjis = []
f = open('./kanji_list.txt', 'r', encoding='utf-8')
line = f.readline().strip()
count = 0
while line:
	kanji = line.split('\t')[0]
	kanjis.append(kanji)
	line = f.readline().strip()
	count+=1
	if count>=800:
		break
f.close()
train_content = ''
test_content = ''

for file in os.listdir(train_folder):
	os.remove(train_folder + file)
for file in os.listdir(test_folder):
	os.remove(test_folder + file)

	
for i in range(N_SAMPLES):
	fontpath = fontpaths[np.random.randint(low=0, high=len(fontpaths))] # <== 这里是宋体路径 
	#fontpath = ""
	font = ImageFont.truetype(fontpath, np.random.randint(low=28, high=33))
	text = ''.join(np.random.choice(kanjis, size=[3]))
	size = font.getsize(text)
	offset = font.getoffset(text)
	while size[0] - offset[0] < CUT_WIDTH:
		text += np.random.choice(kanjis)
		size = font.getsize(text)
		offset = font.getoffset(text)
	text = text[:-1]
	size = font.getsize(text)
	offset = font.getoffset(text)
	lens = []
	for j in range(len(text)):
		lens.append(font.getsize(text[:j+1])[0] - offset[0])
	img_pil = Image.fromarray(np.zeros((size[1] - offset[1],size[0] - offset[0],3),np.uint8))
	draw = ImageDraw.Draw(img_pil)
	draw.text((-offset[0], -offset[1]), text, font = font, fill = (b, g, r, a))
	img = np.array(img_pil)
	if np.random.uniform()>0.1:
		file_name = "{:06d}.png".format(i)
		cv2.imwrite(train_folder + file_name, img)
		train_content += file_name + '\t' + text + '\t' + str(lens) + '\n'
		print(train_folder + file_name)
	else:
		file_name = "{:06d}.png".format(i)
		cv2.imwrite(test_folder + file_name, img)
		test_content += file_name + '\t' + text + '\t' + str(lens) + '\n'
		print(test_folder + file_name)
f = open('./train_data.txt', 'w', encoding='utf-8')
f.write(train_content)
f.close()
f = open('./test_data.txt', 'w', encoding='utf-8')
f.write(test_content)
f.close()