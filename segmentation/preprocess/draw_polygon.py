import tkinter as Tk
from PIL import Image, ImageTk
import os
import random

class MyApp:
	def __init__(self, parent, image_folder='./image/', text_folder='./mark/'):
		self.resize_rate = 0.5
		self.root = parent
		self.root.title("Main frame")
		self.image_folder = image_folder
		self.text_folder = text_folder

		frame1 = Tk.Frame(parent)
		frame1.pack()
		
		root.bind('<Key>', self.onKeyPress)

		self.canvas = Tk.Canvas(frame1, height=250, width=300)
		self.canvas.pack()
		self.canvas.bind('<B1-Motion>', self.onLeftMouseMove) 
		self.canvas.bind('<ButtonPress-1>', self.onLeftMousePress)
		self.canvas.bind('<Double-Button-1>', self.onLeftMouseDoublePress)
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

		self.polygons = [[]]
		self.current_polygon = []

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
			self.canvas.delete('all')
			self.polygons = [[]]
			self.current_polygon = []
			self.current_id-=1
			print('Load File',self.image_list[self.current_id], )
			image = Image.open(self.image_list[self.current_id])
			image = image.resize((int(image.width*self.resize_rate), int(image.height*self.resize_rate)))
			self.photo = ImageTk.PhotoImage(image)
			self.canvas.create_image(0,0,image=self.photo, anchor=Tk.NW)
			self.canvas.config(width=self.photo.width(), height=self.photo.height())
			
			text_name = self.image_list[self.current_id]
			text_name = text_name[text_name.rfind('/') + 1:]
			text_name = text_name + '.txt'
			if text_name in os.listdir(self.text_folder):
				self.loadText(self.text_folder + text_name)

	def moveNext(self):
		if self.current_id<len(self.image_list)-1:
			self.canvas.delete('all')
			self.polygons = [[]]
			self.current_polygon = []
			self.current_id+=1
			print('Load File',self.image_list[self.current_id])
			image = Image.open(self.image_list[self.current_id])
			image = image.resize((int(image.width*self.resize_rate), int(image.height*self.resize_rate)))
			self.photo = ImageTk.PhotoImage(image)
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
		for polygon in self.polygons:
			if len(polygon)>0:
				for line in polygon:
					pos = self.canvas.coords(line)
					s += str(pos[0]/self.resize_rate) + ' ' + str(pos[1]/self.resize_rate) + ' '
				s += '\n'
		if len(s)>0:
			s=s[:-1]
		f.write(s)
		f.close()

	def loadText(self, text_file):
		f = open(text_file, 'r')
		self.polygons = []
		self.current_polygon = []
		s = f.readline().strip()
		while s!='':
			numbers = s.split(' ')
			self.current_polygon = []
			for i in range(0,len(numbers),2):
				self.current_polygon.append((float(numbers[i])*self.resize_rate, float(numbers[i+1])*self.resize_rate))
			polygon = []
			for i in range(len(self.current_polygon)):
				line = self.canvas.create_line(
					self.current_polygon[i%len(self.current_polygon)][0], 
					self.current_polygon[i%len(self.current_polygon)][1], 
					self.current_polygon[(i+1)%len(self.current_polygon)][0], 
					self.current_polygon[(i+1)%len(self.current_polygon)][1], 
					width=1, 
					dash=(5,5),
					fill='red')
				polygon.append(line)
			self.polygons.append(polygon)	
			self.current_polygon = []
			s = f.readline().strip()
			self.polygons.append([])
		f.close()
	
	def onLeftMousePress(self, e):
		if len(self.current_polygon)==0:
			self.current_polygon.append((e.x, e.y))
		else:
			if e.x!=self.current_polygon[-1][0] and e.y!=self.current_polygon[-1][1]:
				self.current_polygon.append((e.x, e.y))
			else:
				return

		if len(self.polygons)>0:
			last_polygon = self.polygons[-1]
			if len(self.current_polygon)>1:
				line = self.canvas.create_line(
					self.current_polygon[-2][0], 
					self.current_polygon[-2][1], 
					self.current_polygon[-1][0], 
					self.current_polygon[-1][1], 
					width=1, 
					dash=(5,5),
					fill='red')
				last_polygon.append(line)

	def onLeftMouseDoublePress(self, e):
		if len(self.current_polygon)>2:
			line = self.canvas.create_line(
				self.current_polygon[-1][0], 
				self.current_polygon[-1][1], 
				self.current_polygon[0][0], 
				self.current_polygon[0][1], 
				width=1, 
				dash=(5,5),
				fill='red')
			last_polygon = self.polygons[-1]
			last_polygon.append(line)
			self.polygons.append([])
			self.current_polygon = []

	def onLeftMouseMove(self, e):
		if len(self.polygons)>0:
			last_polygon = self.polygons[-1]
			if len(last_polygon)>0:
				self.canvas.delete(last_polygon[-1])
				last_polygon.pop()
				self.current_polygon.pop()
				self.current_polygon.append((e.x, e.y))
				if len(self.current_polygon)>1:
					line = self.canvas.create_line(
						self.current_polygon[-2][0], 
						self.current_polygon[-2][1], 
						self.current_polygon[-1][0], 
						self.current_polygon[-1][1], 
						width=1, 
						dash=(5,5),
						fill='red')
					last_polygon.append(line)
	
	def onRightMousePress(self, e):
		self.current_polygon = []
		if len(self.polygons)>0:
			last_polygon = self.polygons[-1]
			
			if len(last_polygon)>0:
				
				for line in last_polygon:
					pos = self.canvas.coords(line)
					self.current_polygon.append((pos[0], pos[1]))
				self.canvas.delete(last_polygon[-1])
				last_polygon.pop()

			else:
				self.polygons.pop()
				if len(self.polygons)==0:
					self.polygons = [[]]

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

	def onKeyPress(self, e):
		if e.keysym=='a':
			self.onKeyPrevPress(e)
		elif e.keysym=='s':
			self.onKeySavePress(e)
		elif e.keysym=='d':
			self.onKeyNextPress(e)
		elif e.keysym=='Escape':
			self.onKeyEscapePress(e)

if __name__ == '__main__':
	root = Tk.Tk()
	image_folder = './ori/'
	text_folder = './mark_polygon/'
	app = MyApp(root, image_folder=image_folder, text_folder=text_folder)
	root.mainloop()