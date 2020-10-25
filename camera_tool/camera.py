import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import datetime
import cv2
import os

cap = cv2.VideoCapture(0)
class MainForm(tk.Frame):
	def __init__(self, parent):
		#parent.resizable(False, False)
		tk.Frame.__init__(self, parent)
		self.pack(fill = tk.BOTH, padx = 8, pady = 8)
		
		self.fr_output_folder = tk.Frame(self)
		self.fr_output_folder.pack(side = tk.TOP, fill = tk.X)
		self.lb_output_folder = tk.Label(self.fr_output_folder, text='Output folder: ')
		self.lb_output_folder.grid(row=0, column = 0)
		self.var_output_folder = tk.StringVar()
		self.ent_output_folder = tk.Entry(self.fr_output_folder, textvariable=self.var_output_folder, state='readonly')
		self.ent_output_folder.grid(row=0, column = 1, padx = 8, sticky = tk.N + tk.S + tk.E + tk.W)
		self.fr_output_folder.columnconfigure(1, weight=1)
		self.btn_output_folder_browser = tk.Button(self.fr_output_folder, text = 'Browser', command=self.choose_output_folder)
		self.btn_output_folder_browser.grid(row=0, column = 2)
		
		self.capture = cv2.VideoCapture(0)
		self.fr_video = tk.Frame(self)
		self.fr_video.pack(side = tk.TOP, fill = tk.X)
		self.canvas = tk.Canvas(self.fr_video, width=640, height=480)
		self.canvas.pack(side = tk.TOP, fill = tk.BOTH, padx = 0, pady = 0)
		self.image = None
		self.display_image = None
		self.frame = None
		self.change_frame()
		
		self.fr_action = tk.Frame(self)
		self.fr_action.pack(side = tk.TOP, fill = tk.X, pady = 8)
		self.btn_save = tk.Button(self.fr_action, text = 'Save', command = self.save)
		self.btn_save.pack(side = tk.LEFT)
		
		# key event
		parent.bind("<KeyPress>", self.handle_key_press_event)
		
	def change_frame(self):
		_, self.image = self.capture.read()
		self.display_image = np.flip(self.image, axis=2)
		self.frame = ImageTk.PhotoImage(image = Image.fromarray(self.display_image))
		self.canvas.create_image(0, 0, image = self.frame, anchor = tk.NW)
		self.canvas.after(100, self.change_frame) # 10 frames/ sec
	
	def choose_output_folder(self):
		output_folder = filedialog.askdirectory(
			initialdir=os.path.dirname(os.path.abspath(__file__)),
			title = 'Select folder')
		if output_folder!='':
			self.var_output_folder.set(output_folder)
			
	def save(self)	:
		if self.var_output_folder.get()!='' and not self.image is None:
			curetime = datetime.datetime.now()
			file_name = '{:04d}'.format(curetime.year) + \
			'{:02d}'.format(curetime.month) + \
			'{:02d}'.format(curetime.day) + \
			'{:02d}'.format(curetime.hour) + \
			'{:02d}'.format(curetime.minute) + \
			'{:02d}'.format(curetime.second) + '.jpg'
			file_path = self.var_output_folder.get() + '/' + file_name
			print('Save', file_path)
			cv2.imwrite(file_path, self.image)
			
	def handle_key_press_event(self, e):
		if e.char=='s':
			self.save()
	
if __name__ == '__main__':
	root = tk.Tk()
	main_form = MainForm(root)
	main_form.mainloop()