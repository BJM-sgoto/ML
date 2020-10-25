from PIL import Image
import numpy as np
import os
source_folder = './cat/'
target_folder = './small_cat/'
for file in os.listdir(source_folder):
	image = Image.open(source_folder + file)
	image = image.resize((128+20,128+20))
	image.save(target_folder + file)
	print(target_folder + file)
	