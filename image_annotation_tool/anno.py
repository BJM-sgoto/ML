import tkinter as tk
from tkinter import filedialog, ttk, PhotoImage
from PIL import Image, ImageTk
import shape
import os
import random

class MainForm(tk.Frame):
	def __init__(self, parent):
		self.parent = parent
		parent.resizable(False, False)
		parent.protocol('WM_DELETE_WINDOW', self.handle_close_window)
		tk.Frame.__init__(self, parent)		
		
		self.pack(fill = tk.BOTH, padx = 8, pady = 8)
		
		self.fr_window_size = tk.Frame(self)
		self.fr_window_size.pack(side = tk.TOP, fill = tk.X)
		self.dm_image_size = ttk.Combobox(self.fr_window_size, state="readonly", values=['400%', '300%', '200%', '100%', '75%', '50%', '25%'])
		self.dm_image_size.current(3)
		self.dm_image_size.pack(side = tk.LEFT)
		
		self.fr_auto_save = tk.Frame(self)
		self.fr_auto_save.pack(side = tk.TOP, fill = tk.X)
		self.var_auto_save = tk.IntVar(0)
		self.btn_auto_save = tk.Checkbutton(self.fr_auto_save, text='Auto Save', variable = self.var_auto_save, offvalue = 0, onvalue = 1)
		self.btn_auto_save.pack(side = tk.LEFT)
				
		self.fr_annotation_choices = tk.Frame(self)
		self.fr_annotation_choices.pack(side = tk.TOP, fill = tk.X)
		self.var_annotation = tk.IntVar(0)
		self.cb_annotation_none = tk.Checkbutton(self.fr_annotation_choices, text='None', variable = self.var_annotation, offvalue = 0, onvalue = 1, command = self.change_annotation_mode)
		self.var_annotation.set(1)
		self.cb_annotation_none.pack(side = tk.LEFT)
		self.cb_annotation_point = tk.Checkbutton(self.fr_annotation_choices, text='Point', variable = self.var_annotation, offvalue = 1, onvalue = 2, command = self.change_annotation_mode)
		self.cb_annotation_point.pack(side = tk.LEFT)
		self.cb_annotation_line = tk.Checkbutton(self.fr_annotation_choices, text='Line', variable = self.var_annotation, offvalue = 1, onvalue = 3, command = self.change_annotation_mode)
		self.cb_annotation_line.pack(side = tk.LEFT)
		self.cb_annotation_rectangle = tk.Checkbutton(self.fr_annotation_choices, text='Rectangle', variable = self.var_annotation, offvalue = 1, onvalue = 4, command = self.change_annotation_mode)
		self.cb_annotation_rectangle.pack(side = tk.LEFT)
		self.cb_annotation_rotated_rectangle = tk.Checkbutton(self.fr_annotation_choices, text='Rotated Rectangle', variable = self.var_annotation, offvalue = 1, onvalue = 5, command = self.change_annotation_mode)
		self.cb_annotation_rotated_rectangle.pack(side = tk.LEFT)
		
		self.fr_input_output_folder = tk.Frame(self)
		self.fr_input_output_folder.pack(side = tk.TOP, fill = tk.X)
		self.lb_input_folder = tk.Label(self.fr_input_output_folder, text='Input folder: ')
		self.lb_output_folder = tk.Label(self.fr_input_output_folder, text='Output folder: ')
		self.lb_input_folder.grid(row = 0, column = 0)
		self.lb_output_folder.grid(row = 1, column = 0)
		self.var_input_folder = tk.StringVar()
		self.ent_input_folder = tk.Entry(self.fr_input_output_folder, textvariable=self.var_input_folder, state='readonly')
		self.var_output_folder = tk.StringVar()
		self.ent_output_folder = tk.Entry(self.fr_input_output_folder, textvariable=self.var_output_folder, state='readonly')
		self.ent_input_folder.grid(row = 0, column = 1, padx = 8, sticky = tk.N + tk.S + tk.E + tk.W)
		self.ent_output_folder.grid(row = 1, column = 1, padx = 8, sticky = tk.N + tk.S + tk.E + tk.W)
		self.fr_input_output_folder.columnconfigure(1, weight=1)
		self.btn_input_folder_browser = tk.Button(self.fr_input_output_folder, text = 'Browser', command=self.choose_input_folder)
		self.btn_output_folder_browser = tk.Button(self.fr_input_output_folder, text = 'Browser', command=self.choose_output_folder)
		self.btn_input_folder_browser.grid(row = 0, column = 2)
		self.btn_output_folder_browser.grid(row = 1, column = 2)
		
		self.fr_files_image = tk.Frame(self)
		self.fr_files_image.pack(side = tk.TOP, fill = tk.BOTH)
		self.fr_files = tk.Frame(self.fr_files_image)
		self.fr_files.grid(row = 0, column = 0, sticky= tk.N + tk.S, pady=(8, 0))
		self.lb_files = tk.Listbox(self.fr_files, selectmode=tk.SINGLE)
		self.lb_files.bind('<Double-Button-1>', self.click_file_item)
		self.fr_input_output_folder.rowconfigure(0, weight=1)
		self.fr_image = tk.Frame(self.fr_files_image, background = 'azure2')
		self.fr_image.grid(row = 0, column = 1, pady = (8, 0), sticky= tk.N + tk.S + tk.E + tk.W)
		self.fr_files_image.columnconfigure(1, weight=1)
		self.canvas = tk.Canvas(self.fr_image, width=640, height=480)
		self.canvas.pack(padx = 0, pady = 0)
		self.canvas.bind('<Button-1>', self.handle_left_mouse_down_on_canvas)
		self.canvas.bind('<Button-3>', self.handle_right_mouse_down_on_canvas)
		self.canvas.bind('<B1-Motion>', self.handle_left_mouse_move_on_canvas)
		self.lb_files.pack()
		
		# prev save next buttons
		self.fr_action_buttons = tk.Frame(self)
		self.fr_action_buttons.pack(side = tk.TOP, pady  =(8,0))
		self.btn_prev = tk.Button(self.fr_action_buttons, text = '  Prev  ', command = self.move_to_prev_image)
		self.btn_prev.pack(side = tk.LEFT, padx = 8)
		self.btn_save = tk.Button(self.fr_action_buttons, text = '  Save  ', command = self.save_text_file)
		self.btn_save.pack(side = tk.LEFT, padx = 8)
		self.btn_next = tk.Button(self.fr_action_buttons, text = '  Next  ', command = self.move_to_next_image)
		self.btn_next.pack(side = tk.LEFT, padx = 8)
		
		# key event
		parent.bind("<KeyPress>", self.handle_key_press_event)
		self.image = None
		
		# data
		self.points = []
		self.lines = []
		self.rectangles = []
		self.rotated_rectangles = []
		self.selected_shape = None
		self.selected_subshape = -1
		
		# load data from cache
		f = open('cache.txt', 'r')
		try:
			input_folder = f.readline().strip()
			output_folder = f.readline().strip()
			f.close()
			if os.path.exists(input_folder) and os.path.isdir(input_folder):
				self.open_input_folder(input_folder)
			if os.path.exists(output_folder) and os.path.isdir(output_folder):
				self.open_output_folder(output_folder)
		except:
			f.close()
	
	def open_input_folder(self, input_folder):
		self.var_input_folder.set(input_folder)
		self.lb_files.delete(0, tk.END)
		files = os.listdir(input_folder)
		files = [file for file in files if file.endswith('jpg') or file.endswith('jpeg') or file.endswith('png') or file.endswith('bmp')]
		for i, file in enumerate(files):
			self.lb_files.insert(i, file)
		if self.lb_files.size()>0:
			self.lb_files.selection_set(0)
			self.lb_files.activate(0)
			self.open_image()
	
	def choose_input_folder(self):
		input_folder = filedialog.askdirectory(
			initialdir = os.path.dirname(os.path.abspath(__file__)),
			title = 'Select folder')
		if input_folder!='':
			self.open_input_folder(input_folder)
			
	def open_output_folder(self, output_folder):
		self.var_output_folder.set(output_folder)
	
	def choose_output_folder(self):
		output_folder = filedialog.askdirectory(
			initialdir=os.path.dirname(os.path.abspath(__file__)),
			title = 'Select folder')
		if output_folder!='':
			self.open_output_folder(output_folder)
	
	def click_file_item(self, event):
		self.open_image()
		
	def open_image(self):
		self.canvas.delete('all')
		self.points.clear()
		self.lines.clear()
		self.rectangles.clear()
		self.rotated_rectangles.clear()
		image_path = self.var_input_folder.get() + '/'
		image_path += self.lb_files.get(self.lb_files.curselection())
		img = Image.open(image_path)
		img = ImageTk.PhotoImage(img)
		self.image = img
		self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
		self.canvas.configure(width=self.image.width(), height=self.image.height())
		self.read_text_file()
		
	def read_text_file(self):
		file_path = self.var_output_folder.get() + '/' 
		if self.var_annotation.get() == 2:
			file_path += 'point/'
			file_path += self.lb_files.get(self.lb_files.curselection()) + '.txt'
			if os.path.exists(file_path) and os.path.isfile(file_path):
				f = open(file_path, 'r')
				s = f.read().strip()
				f.close()
				ss = s.split('\n')
				for str_point in ss:
					point_elements = str_point.split()
					self.selected_shape, self.selected_subshape = shape.Point.create_object_with_data(self.canvas, float(point_elements[0]), float(point_elements[1]))
					self.points.append(self.selected_shape)
				# select last point	
				if self.selected_shape!=None:
					self.selected_shape.select()
		elif self.var_annotation.get() == 3: # line
			file_path += 'line/'
			file_path += self.lb_files.get(self.lb_files.curselection()) + '.txt'
			if os.path.exists(file_path) and os.path.isfile(file_path):
				f = open(file_path, 'r')
				s = f.read().strip()
				f.close()
				ss = s.split('\n')
				for str_line in ss:
					line_elements = str_line.split()
					self.selected_shape, self.selected_subshape = shape.Line.create_object_with_data(self.canvas, float(line_elements[0]), float(line_elements[1]), float(line_elements[2]), float(line_elements[3]))
					self.lines.append(self.selected_shape)
				# select last line
				if self.selected_shape!=None:
					self.selected_shape.select()
		elif self.var_annotation.get() == 4: # rectangle
			file_path += 'rectangle/'
			file_path += self.lb_files.get(self.lb_files.curselection()) + '.txt'
			if os.path.exists(file_path) and os.path.isfile(file_path):
				f = open(file_path, 'r')
				s = f.read().strip()
				f.close()
				ss = s.split('\n')
				for str_rectangle in ss:
					rectangle_elements = str_rectangle.split()
					self.selected_shape, self.selected_subshape = shape.Rectangle.create_object_with_data(self.canvas, float(rectangle_elements[0]), float(rectangle_elements[1]), float(rectangle_elements[2]), float(rectangle_elements[3]))
					self.rectangles.append(self.selected_shape)
				# select last rectangle
				if self.selected_shape!=None:
					self.selected_shape.select()
		elif self.var_annotation.get() == 5: # rotated rectangle
			file_path += 'rotated_rectangle/'
			file_path += self.lb_files.get(self.lb_files.curselection()) + '.txt'
			if os.path.exists(file_path) and os.path.isfile(file_path):
				print(file_path)
				f = open(file_path, 'r')
				s = f.read().strip()
				f.close()
				ss = s.split('\n')
				for str_rotated_rectangle in ss:
					rotated_rectangle_elements = str_rotated_rectangle.split()
					self.selected_shape, self.selected_subshape = shape.RotatedRectangle.create_object_with_data(self.canvas, float(rotated_rectangle_elements[0]),float(rotated_rectangle_elements[1]),float(rotated_rectangle_elements[2]),float(rotated_rectangle_elements[3]),float(rotated_rectangle_elements[4]))
					self.rotated_rectangles.append(self.selected_shape)
				# select last rotated rectangle
				if self.selected_shape!=None:
					self.selected_shape.select()
				
	def handle_key_press_event(self, event):
		if self.lb_files.size()>0:
			if event.char=='a':
				self.move_to_prev_image()
			elif event.char=='s':
				self.save_text_file()
			elif event.char=='d':
				self.move_to_next_image()
			else: # special key
				keysym = event.keysym.lower()
				last_obj = None
				if self.var_annotation.get()==2 and len(self.points)>0:
					last_obj = self.points[-1]
				elif self.var_annotation.get()==3 and len(self.lines)>0:
					last_obj = self.lines[-1]
				elif self.var_annotation.get()==4 and len(self.rectangles)>0:
					last_obj = self.rectangles[-1]
				if not last_obj is None:
					if keysym=='up':
						last_obj.move(0, -1)
					elif keysym=='down':
						last_obj.move(0, 1)
					elif keysym=='left':
						last_obj.move(-1, 0)
					elif keysym=='right':
						last_obj.move(1, 0)
		
	def move_to_next_image(self):
		if self.lb_files.curselection()[0] < self.lb_files.size() - 1:
			if self.var_auto_save.get()==1:
				self.save_text_file()
			cur_selection = self.lb_files.curselection()[0]
			self.lb_files.selection_clear(0, tk.END)
			self.lb_files.selection_set(cur_selection + 1)
			self.lb_files.activate(cur_selection + 1)
			self.open_image()
	
	def save_text_file(self):
		# save before remove
		code = self.lb_files.get(self.lb_files.curselection())
		if self.var_annotation.get()==2: # point
			point_folder = self.var_output_folder.get() + '/point/'
			if not os.path.exists(point_folder):
				os.mkdir(point_folder)
			f = open(point_folder + code + '.txt', 'w')
			for point in self.points:
				f.write(point.to_string() + '\n')
			f.close()
		elif self.var_annotation.get()==3: # line
			line_folder = self.var_output_folder.get() + '/line/'
			if not os.path.exists(line_folder):
				os.mkdir(line_folder)
			f = open(line_folder + code + '.txt', 'w')
			for line in self.lines:
				f.write(line.to_string() + '\n')
			f.close()
		elif self.var_annotation.get()==4: # rectangle
			rectangle_folder = self.var_output_folder.get() + '/rectangle/'
			if not os.path.exists(rectangle_folder):
				os.mkdir(rectangle_folder)
			f = open(rectangle_folder + code + '.txt', 'w')
			for rectangle in self.rectangles:
				f.write(rectangle.to_string() + '\n')
			f.close()
		elif self.var_annotation.get()==5: # rotated rectangle
			rotated_rectangle_folder = self.var_output_folder.get() + '/rotated_rectangle/'
			if not os.path.exists(rotated_rectangle_folder):
				os.mkdir(rotated_rectangle_folder)
			f = open(rotated_rectangle_folder + code + '.txt', 'w')
			for rotated_rectangle in self.rotated_rectangles:
				f.write(rotated_rectangle.to_string() + '\n')
			f.close()
			
	def move_to_prev_image(self):
		if self.lb_files.curselection()[0] > 0:
			if self.var_auto_save.get()==1:
				self.save_text_file()
			cur_selection = self.lb_files.curselection()[0]
			self.lb_files.selection_clear(0, tk.END)
			self.lb_files.selection_set(cur_selection - 1)
			self.lb_files.activate(cur_selection - 1)
			self.open_image()
			
	def handle_left_mouse_down_on_canvas(self, event):
		x = event.x
		y = event.y
		# check if there is any close object
		close_objects = self.canvas.find_overlapping(x - 3, y - 3, x + 3, y + 3)
		random.shuffle(list(close_objects))
		selected_shape = None
		selected_subshape = -1
		for close_object in close_objects:
			if self.var_annotation.get()==2: # point
				for shape_object in self.points:
					if shape_object.has_subshape(close_object):
						selected_shape = shape_object
						selected_subshape = close_object
			elif self.var_annotation.get()==3: # line
				for shape_object in self.lines:
					if shape_object.has_subshape(close_object):
						selected_shape = shape_object
						selected_subshape = close_object
			elif self.var_annotation.get()==4: # rectangle 
				for shape_object in self.rectangles:
					if shape_object.has_subshape(close_object):
						selected_shape = shape_object
						selected_subshape = close_object
			elif self.var_annotation.get() == 5: # rotated rectangle 
				for shape_object in self.rotated_rectangles:
					if shape_object.has_subshape(close_object):
						selected_shape = shape_object
						selected_subshape = close_object
			if selected_shape!=None:
				break
		if selected_shape!=None:
			if self.selected_shape!=None:
				self.selected_shape.unselect()
			self.selected_shape = selected_shape
			self.selected_subshape = selected_subshape
			self.selected_shape.select()
		# other wise
		else:
			if self.selected_shape!=None:
				self.selected_shape.unselect()
				self.selected_shape = None
				self.selected_subshape = -1
			else:
				if self.var_annotation.get()==2: # point
					self.selected_shape, self.selected_subshape = shape.Point.create_object(self.canvas, x, y)
					self.selected_shape.select()
					self.points.append(self.selected_shape)
				elif self.var_annotation.get()==3: # line
					self.selected_shape, self.selected_subshape = shape.Line.create_object(self.canvas, x, y)
					self.selected_shape.select()
					self.lines.append(self.selected_shape)
				elif self.var_annotation.get()==4: # rectangle 
					self.selected_shape, self.selected_subshape = shape.Rectangle.create_object(self.canvas, x, y)
					self.selected_shape.select()
					self.rectangles.append(self.selected_shape)
				elif self.var_annotation.get() == 5:
					self.selected_shape, self.selected_subshape = shape.RotatedRectangle.create_object(self.canvas, x, y)
					self.selected_shape.select()
					self.rotated_rectangles.append(self.selected_shape)
			
	def handle_left_mouse_move_on_canvas(self, event):
		if self.selected_shape!=None: # if something is selected
			self.selected_shape.pull_subshape(self.selected_subshape, event.x, event.y)
		
	def handle_right_mouse_down_on_canvas(self, event):
		# unselect object
		if not self.selected_shape is None:
			self.selected_shape.unselect()
			self.selected_shape = None
			self.selected_subshape = -1
		else:
			if self.var_annotation.get()==2: # point
				if len(self.points)>0: 
					point = self.points.pop()
					point.remove()	
			elif self.var_annotation.get()==3:
				if len(self.lines)>0:
					line = self.lines.pop()
					line.remove()
			elif self.var_annotation.get()==4:
				if len(self.rectangles)>0:
					rectangle = self.rectangles.pop()
					rectangle.remove()
			elif self.var_annotation.get()==5:
				if len(self.rotated_rectangles)>0:
					rotated_rectangle = self.rotated_rectangles.pop()
					rotated_rectangle.remove()
					
	def handle_close_window(self):
		input_folder = self.var_input_folder.get()
		output_folder = self.var_output_folder.get()
		f = open('cache.txt', 'w')
		f.write(input_folder + '\n')
		f.write(output_folder)
		f.close()
		self.parent.destroy()
				
	def change_annotation_mode(self):
		if self.selected_shape!=None:
			self.selected_shape.unselect()
			self.selected_shape = None
			self.selected_subshape = -1
			
		for point in self.points:
			point.remove()
		for line in self.lines:
			line.remove()
		for rectangle in self.rectangles:
			rectangle.remove()
		for rotated_rectangle in self.rotated_rectangles:
			rotated_rectangle.remove()
		
		self.points.clear()
		self.lines.clear()
		self.rectangles.clear()
		self.rotated_rectangles.clear()
			
		if self.lb_files.size()>0:
			self.read_text_file()
			
if __name__ == '__main__':
	root = tk.Tk()
	main_form = MainForm(root)
	main_form.mainloop()