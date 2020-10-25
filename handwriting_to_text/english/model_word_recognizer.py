import numpy as np

WORD_HEIGHT = 46

class Model:
	def __init__(self):
		
	
	def make_dataset(self, image_folder, text_file):
		f = open(text_file, 'r')
		line = f.readline().strip()
		while line:
			elements = line.split('\t')
			image_path = image_folder + elements[0]
			
			line = f.readline().strip()
		f.close()
		
	def shuffle_dataset(self, dataset):
		
	
	def process_batch(self, raw_batch):
		
	
	def encode(self, images):