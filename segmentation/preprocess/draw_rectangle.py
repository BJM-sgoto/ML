import tkinter as Tk
from PIL import Image, ImageTk
import os
import random

class MyApp:
	def __init__(self, parent, image_folder='./image/', text_folder='./mark/'):
		self.root = parent
		self.root.title("Main frame")
		self.image_folder = image_folder
		self.text_folder = text_folder

		frame1 = Tk.Frame(parent)
		frame1.pack()

		root.bind('<Key-Escape>', self.onKeyEscapePress)
		root.bind('<Key-A>', self.onKeyPrevPress)
		root.bind('<Key-S>', self.onKeySavePress)
		root.bind('<Key-D>', self.onKeyNextPress)

		self.canvas = Tk.Canvas(frame1, height=250, width=300)
		self.canvas.pack()
		self.canvas.bind('<B1-Motion>', self.onLeftMouseMove) 
		self.canvas.bind('<ButtonPress-1>', self.onLeftMousePress)
		self.canvas.bind('<ButtonPress-3>', self.onRightMousePress)		

		frame2 = Tk.Frame(parent)
		frame2.pack(fill=Tk.X, expand=1)
		
		self.btn_prev = Tk.Button(frame2, text="Prev", command=self.movePrev)
		self.btn_prev.pack(side=Tk.LEFT, fill=Tk.X, expand=0.3)

		self.btn_save = Tk.Button(frame2, text="Save", command=self.saveText)
		self.btn_save.pack(side=Tk.LEFT, fill=Tk.X, expand=0.3)

		self.btn_next = Tk.Button(frame2, text="Next", command=self.moveNext)
		self.btn_next.pack(side=Tk.LEFT, fill=Tk.X, expand=0.3)

		self.loadData(image_folder=image_folder, text_folder=text_folder)

		self.current_id = -1
		self.moveNext()

		self.area_size = 192
		self.rectangles = []

	def loadData(self,image_folder, text_folder):
		self.image_list = os.listdir(image_folder)
		text_list = os.listdir(text_folder)
		text_codes = []
		for text in text_list:
			code = text[:text.rfind('.')]
			if code in self.image_list:
				pos = self.image_list.index(code)
				self.image_list.pop(pos)
		random.shuffle(self.image_list)
		for i in range(len(self.image_list)):
			self.image_list[i] = image_folder + self.image_list[i]


	def movePrev(self):
		if self.current_id>0:
			self.rectangles = []
			self.current_id-=1
			print('Load File',self.image_list[self.current_id])
			self.photo = ImageTk.PhotoImage(file=self.image_list[self.current_id])
			self.canvas.create_image(0,0,image=self.photo, anchor=Tk.NW)
			self.canvas.config(width=self.photo.width(), height=self.photo.height())
			
			text_name = self.image_list[self.current_id]
			text_name = text_name[text_name.rfind('/') + 1:]
			text_name = text_name + '.txt'
			if text_name in os.listdir(self.text_folder):
				self.loadText(self.text_folder + text_name)

	def saveText(self):
		text_name = self.image_list[self.current_id]
		text_name = text_name[text_name.rfind('/') + 1:]
		text_name = self.text_folder + text_name + '.txt'
		print('Save file', text_name)
		f = open(text_name, 'w')
		s = ''
		for rect in self.rectangles:
			pos = self.canvas.coords(rect)
			s += str(pos[0]) + ' ' + str(pos[1]) + ' ' + str(pos[2]) + ' ' + str(pos[3]) + '\n'
		if len(s)>0:
			s = s[:-1]
		f.write(s)
		f.close()

	def loadText(self, text_file):
		f = open(text_file, 'r')
		self.rectangles = []
		s = f.readline().strip()
		while s!='':
			numbers = s.split(' ')
			rect = self.canvas.create_rectangle(float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3]))
			self.rectangles.append(rect)
			s = f.readline().strip()
		f.close()

	def moveNext(self):
		if self.current_id<len(self.image_list)-1:
			self.rectangles = []
			self.current_id+=1
			print('Load File',self.image_list[self.current_id])
			self.photo = ImageTk.PhotoImage(file=self.image_list[self.current_id])
			self.canvas.create_image(0,0,image=self.photo, anchor=Tk.NW)
			self.canvas.config(width=self.photo.width(), height=self.photo.height())

			text_name = self.image_list[self.current_id]
			text_name = text_name[text_name.rfind('/') + 1:]
			text_name = text_name + '.txt'
			if text_name in os.listdir(self.text_folder):
				self.loadText(self.text_folder + text_name)
			
	def onKeyEscapePress(self, e):
		#
		self.root.quit()

	def onKeyPrevPress(self, e):
		#
		self.movePrev()

	def onKeySavePress(self, e):
		#
		self.saveText()

	def onKeyNextPress(self, e):
		#
		self.moveNext()

	def onLeftMouseMove(self, e):
		start_x = e.x - int(self.area_size/2)
		start_y = e.y - int(self.area_size/2)
		if start_x<0:
			start_x = 0
		if start_y<0:
			start_y = 0
		end_x = start_x + self.area_size
		end_y = start_y + self.area_size
		if end_x>self.photo.width():
			end_x = self.photo.width()
			start_x = end_x - self.area_size
		if end_y>self.photo.height():
			end_y = self.photo.height()
			start_y = end_y - self.area_size
		if len(self.rectangles)>0:
			rect = self.rectangles[-1]
			pos = self.canvas.coords(rect)
			self.canvas.move(rect, start_x - pos[0], start_y - pos[1])
		else:
			rect = self.canvas.create_rectangle([start_x, start_y, end_x, end_y])
		
	def onLeftMousePress(self, e):
		start_x = e.x - int(self.area_size/2)
		start_y = e.y - int(self.area_size/2)
		if start_x<0:
			start_x = 0
		if start_y<0:
			start_y = 0
		end_x = start_x + self.area_size
		end_y = start_y + self.area_size
		if end_x>self.photo.width():
			end_x = self.photo.width()
			start_x = end_x - self.area_size
		if end_y>self.photo.height():
			end_y = self.photo.height()
			start_y = end_y - self.area_size
		rect = self.canvas.create_rectangle([start_x, start_y, end_x, end_y])
		self.rectangles.append(rect)

	def onRightMousePress(self, e):
		if len(self.rectangles)>0:
			self.canvas.delete(self.rectangles[-1])
			self.rectangles.pop(-1)

if __name__ == '__main__':
	root = Tk.Tk()
	image_folder = './image/'
	text_folder = './mark/'
	app = MyApp(root, image_folder=image_folder, text_folder=text_folder)
	root.mainloop()