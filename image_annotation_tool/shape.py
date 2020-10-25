import math

class Shape:

	COLOR_SELECTED = 'red'
	COLOR_UNSELECTED = 'blue'

	def __init__(self, canvas, sub_shapes):
		self.canvas = canvas
		self.sub_shapes = sub_shapes
		self.selected = False
		
	def move(self, dx, dy):
		for sub_shape in self.sub_shapes:
			self.canvas.move(sub_shape, dx, dy)
		
	def rotate(self, d_angle):
		pass
		
	def resize(self, new_size):
		pass
		
	def remove(self):
		for sub_shape in self.sub_shapes:
			self.canvas.delete(sub_shape)
			
	@staticmethod
	def create_object(canvas, x, y):
		pass
			
	def to_string(self):
		pass
		
	def change_color(self, color):
		for sub_shape in self.sub_shapes:
			self.canvas.itemconfig(sub_shape, fill=color)
			
	# only check with point
	def has_subshape(self, subshape):
		pass
		
	def select(self):
		self.selected = True
		self.change_color(Shape.COLOR_SELECTED)
	
	def unselect(self):
		self.selected = False
		self.change_color(Shape.COLOR_UNSELECTED)
		
	def pull_subshape(self, subshape, x, y):
		pass
		
class Point(Shape):

	_points_pos = [0]

	def __init__(self, canvas, sub_shapes):
		super(Point, self).__init__(canvas, sub_shapes)
		
	@staticmethod
	def create_object(canvas, x, y):
		point = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		return Point(canvas, [point]), point
		
	@staticmethod
	def create_object_with_data(canvas, x, y):
		point = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		return Point(canvas, [point]), point
	
	def has_subshape(self, subshape):
		for pos in Point._points_pos:
			if self.sub_shapes[pos] == subshape:
				return True
		return False
		
	def to_string(self):
		point_coord = self.canvas.coords(self.sub_shapes[0])
		return str(point_coord[0] + 3) + ' ' + str(point_coord[1] + 3)
		
	def pull_subshape(self, subshape, x, y):
		point = self.sub_shapes[0]
		if subshape==point:
			self.canvas.coords(point, (x-3, y-3, x+3, y+3))
		
class Line(Shape):

	_points_pos = [0, 1]
	
	def __init__(self, canvas, sub_shapes):
		super(Line, self).__init__(canvas, sub_shapes)
	
	@staticmethod
	def create_object(canvas, x, y):
		point1 = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		point2 = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		line = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		return Line(canvas, [point1, point2, line]), point2
		
	@staticmethod
	def create_object_with_data(canvas, x1, y1, x2, y2):
		point1 = canvas.create_rectangle(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill = Shape.COLOR_UNSELECTED)
		point2 = canvas.create_rectangle(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill = Shape.COLOR_UNSELECTED)
		line = canvas.create_line(x1, y1, x2, y2, fill = Shape.COLOR_UNSELECTED)
		return Line(canvas, [point1, point2, line]), point2
		
	def has_subshape(self, subshape):
		for pos in Line._points_pos:
			if self.sub_shapes[pos] == subshape:
				return True
		return False
		
	def to_string(self):
		line_coord = self.canvas.coords(self.sub_shapes[2])
		return str(line_coord[0]) + ' ' + str(line_coord[1]) + ' ' + str(line_coord[2]) + ' ' + str(line_coord[3])
		
	def pull_subshape(self, subshape, x, y):
		point1 = self.sub_shapes[0]
		point2 = self.sub_shapes[1]
		line = self.sub_shapes[2]
		if subshape==point1:
			self.canvas.coords(point1, (x-3, y-3, x+3, y+3))
			point2_coord = self.canvas.coords(point2)
			self.canvas.coords(line, (x, y, point2_coord[0] + 3, point2_coord[1] + 3))
		elif subshape == point2:
			self.canvas.coords(point2, (x-3, y-3, x+3, y+3))
			point1_coord = self.canvas.coords(point1)
			self.canvas.coords(line, (point1_coord[0] + 3, point1_coord[1] + 3, x, y))
		else:
			print('Line error: pull')
		
class Rectangle(Shape):

	_points_pos = [0, 1, 2, 3, 8]

	def __init__(self, canvas, sub_shapes):
		self.angle = 0
		super(Rectangle, self).__init__(canvas, sub_shapes)
		
	@staticmethod
	def create_object(canvas, x, y):
		point1 = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		point2 = canvas.create_rectangle(x - 2, y - 3, x + 4, y + 3, fill = Shape.COLOR_UNSELECTED)
		point3 = canvas.create_rectangle(x - 2, y - 2, x + 4, y + 4, fill = Shape.COLOR_UNSELECTED)
		point4 = canvas.create_rectangle(x - 3, y - 2, x + 3, y + 4, fill = Shape.COLOR_UNSELECTED)
		line1 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		line2 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		line3 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		line4 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		center = canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		return  Rectangle(canvas, [point1, point2, point3, point4, line1, line2, line3, line4, center]), point3
	
	@staticmethod
	def create_object_with_data(canvas, center_x, center_y, width, height):
		half_width = width / 2
		half_height = height / 2
		x1 = center_x - half_width
		y1 = center_y - half_height
		x2 = center_x + half_width
		y2 = center_y - half_height
		x3 = center_x + half_width
		y3 = center_y + half_height
		x4 = center_x - half_width
		y4 = center_y + half_height
		
		point1 = canvas.create_rectangle(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill = Shape.COLOR_UNSELECTED)
		point2 = canvas.create_rectangle(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill = Shape.COLOR_UNSELECTED)
		point3 = canvas.create_rectangle(x3 - 3, y3 - 3, x3 + 3, y3 + 3, fill = Shape.COLOR_UNSELECTED)
		point4 = canvas.create_rectangle(x4 - 3, y4 - 3, x4 + 3, y4 + 3, fill = Shape.COLOR_UNSELECTED)
		line1 = canvas.create_line(x1, y1, x2, y2, fill = Shape.COLOR_UNSELECTED)
		line2 = canvas.create_line(x2, y2, x3, y3, fill = Shape.COLOR_UNSELECTED)
		line3 = canvas.create_line(x3, y3, x4, y4, fill = Shape.COLOR_UNSELECTED)
		line4 = canvas.create_line(x4, y4, x1, y1, fill = Shape.COLOR_UNSELECTED)
		center = canvas.create_oval(center_x - 3, center_y - 3, center_x + 3, center_y + 3, fill = Shape.COLOR_UNSELECTED)
		return  Rectangle(canvas, [point1, point2, point3, point4, line1, line2, line3, line4, center]), point3
	
	def has_subshape(self, subshape):
		for pos in Rectangle._points_pos:
			if self.sub_shapes[pos] == subshape:
				return True
		return False
		
	def to_string(self):
		corner1_coord = self.canvas.coords(self.sub_shapes[0])
		corner2_coord = self.canvas.coords(self.sub_shapes[1])
		corner3_coord = self.canvas.coords(self.sub_shapes[2])
		width = math.sqrt((corner1_coord[0]-corner2_coord[0])**2 + (corner1_coord[1]-corner2_coord[1])**2)
		height = math.sqrt((corner3_coord[0]-corner2_coord[0])**2 + (corner3_coord[1]-corner2_coord[1])**2)
		center_coord = self.canvas.coords(self.sub_shapes[8])
		print('width', width, 'height', height)
		return str(center_coord[0] + 3) + ' ' + str(center_coord[1] + 3) + ' ' + str(width) + ' ' + str(height)
		
	def pull_subshape(self, subshape, x, y):
		corner1 = self.sub_shapes[0]
		corner2 = self.sub_shapes[1]
		corner3 = self.sub_shapes[2]
		corner4 = self.sub_shapes[3]
		line1 = self.sub_shapes[4]
		line2 = self.sub_shapes[5]
		line3 = self.sub_shapes[6]
		line4 = self.sub_shapes[7]
		center = self.sub_shapes[8]
		if subshape == corner1 or subshape == corner2 or subshape==corner3 or subshape == corner4:
			if subshape == corner1:
				point1 = corner3
				point2 = corner4
				point3 = corner1
				point4 = corner2
				
				n_line1 = line3
				n_line2 = line4
				n_line3 = line1
				n_line4 = line2
				
				sin_angle = math.sin(self.angle + math.pi)
				cos_angle = math.cos(self.angle + math.pi)
			elif subshape == corner2:
				point1 = corner4
				point2 = corner1
				point3 = corner2
				point4 = corner3
				
				n_line1 = line4
				n_line2 = line1
				n_line3 = line2
				n_line4 = line3
				
				sin_angle = math.sin(self.angle + math.pi/2)
				cos_angle = math.cos(self.angle + math.pi/2)
			elif subshape == corner3:
				point1 = corner1
				point2 = corner2
				point3 = corner3
				point4 = corner4
				
				n_line1 = line1
				n_line2 = line2
				n_line3 = line3
				n_line4 = line4
				
				sin_angle = math.sin(self.angle)
				cos_angle = math.cos(self.angle)
			elif subshape == corner4:
				point1 = corner2
				point2 = corner3
				point3 = corner4
				point4 = corner1
				
				n_line1 = line2
				n_line2 = line3
				n_line3 = line4
				n_line4 = line1
				
				sin_angle = math.sin(self.angle + math.pi*3/2)
				cos_angle = math.cos(self.angle + math.pi*3/2)
			# point 1 -> 4: clockwise
			point1_coord = self.canvas.coords(point1)
			point2_coord = self.canvas.coords(point2)
			point3_coord = self.canvas.coords(point3)
			point4_coord = self.canvas.coords(point4)
			
			new_d12 = (x - point1_coord[0] - 3)*cos_angle + (y - point1_coord[1] - 3) * sin_angle
			new_x2 = point1_coord[0] + 3 + new_d12 * cos_angle
			new_y2 = point1_coord[1] + 3 + new_d12 * sin_angle
			new_d14 = -(x - point1_coord[0] - 3) * sin_angle + (y - point1_coord[1] - 3) * cos_angle
			new_x4 = point1_coord[0] + 3 - new_d14 * sin_angle
			new_y4 = point1_coord[1] + 3 + new_d14 * cos_angle #???
			self.canvas.coords(point2, (new_x2 - 3, new_y2 - 3, new_x2 + 3, new_y2 + 3))
			self.canvas.coords(point3, (x - 3, y - 3, x + 3, y + 3))
			self.canvas.coords(point4, (new_x4 - 3, new_y4 - 3, new_x4 + 3, new_y4 + 3))
			center = self.sub_shapes[8]
			new_center_x = (point1_coord[0] + 3 + x)/2
			new_center_y = (point1_coord[1] + 3 + y)/2
			self.canvas.coords(center, (new_center_x - 3, new_center_y - 3, new_center_x + 3, new_center_y + 3))
			self.canvas.coords(n_line1, (point1_coord[0] + 3, point1_coord[1] + 3, new_x2, new_y2))
			self.canvas.coords(n_line2, (new_x2, new_y2, x, y))
			self.canvas.coords(n_line3, (x, y, new_x4, new_y4))
			self.canvas.coords(n_line4, (new_x4, new_y4, point1_coord[0] + 3, point1_coord[1] + 3))
		elif subshape==center:
			center_coord = self.canvas.coords(center)
			self.move(x - center_coord[0] - 3, y - center_coord[1] - 3)
			
class RotatedRectangle(Rectangle):

	_points_pos = [0, 1, 2, 3, 8, 10]

	def __init__(self, canvas, sub_shapes):
		super(RotatedRectangle, self).__init__(canvas, sub_shapes)
		lever = sub_shapes[9]
		lever_coord = canvas.coords(lever)
		dx = lever_coord[2] - lever_coord[0]
		dy = lever_coord[3] - lever_coord[1]
		d = math.sqrt(dx**2 + dy**2)
		if d==0:
			self.angle=0
		else:
			sin_angle = dy/d
			cos_angle = dx/d
			self.angle = math.acos(cos_angle)
			if sin_angle<0:
				self.angle = 2 * math.pi - self.angle
		
		
	@staticmethod
	def create_object(canvas, x, y):
		point1 = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		point2 = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		point3 = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		point4 = canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		line1 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		line2 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		line3 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		line4 = canvas.create_line(x, y, x, y, fill = Shape.COLOR_UNSELECTED)
		center = canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		lever = canvas.create_line(x, y, x + 40, y, dash=(4,4), fill = Shape.COLOR_UNSELECTED)
		lever_end = canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill = Shape.COLOR_UNSELECTED)
		return RotatedRectangle(canvas, [point1, point2, point3, point4, line1, line2, line3, line4, center, lever, lever_end]), point3
		
	@staticmethod
	def create_object_with_data(canvas, center_x, center_y, width, height, angle):
		half_width = width / 2
		half_height = height / 2
		sin_angle = math.sin(angle)
		cos_angle = math.cos(angle)
		x1 = center_x - half_width * cos_angle + half_height * sin_angle
		y1 = center_y - half_width * sin_angle - half_height * cos_angle
		x2 = center_x + half_width * cos_angle + half_height * sin_angle
		y2 = center_y + half_width * sin_angle - half_height * cos_angle
		x3 = center_x + half_width * cos_angle - half_height * sin_angle
		y3 = center_y + half_width * sin_angle + half_height * cos_angle
		x4 = center_x - half_width * cos_angle - half_height * sin_angle
		y4 = center_y - half_width * sin_angle + half_height * cos_angle
		
		point1 = canvas.create_rectangle(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill = Shape.COLOR_UNSELECTED)
		point2 = canvas.create_rectangle(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill = Shape.COLOR_UNSELECTED)
		point3 = canvas.create_rectangle(x3 - 3, y3 - 3, x3 + 3, y3 + 3, fill = Shape.COLOR_UNSELECTED)
		point4 = canvas.create_rectangle(x4 - 3, y4 - 3, x4 + 3, y4 + 3, fill = Shape.COLOR_UNSELECTED)
		line1 = canvas.create_line(x1, y1, x2, y2, fill = Shape.COLOR_UNSELECTED)
		line2 = canvas.create_line(x2, y2, x3, y3, fill = Shape.COLOR_UNSELECTED)
		line3 = canvas.create_line(x3, y3, x4, y4, fill = Shape.COLOR_UNSELECTED)
		line4 = canvas.create_line(x4, y4, x1, y1, fill = Shape.COLOR_UNSELECTED)
		center = canvas.create_oval(center_x - 3, center_y - 3, center_x + 3, center_y + 3, fill = Shape.COLOR_UNSELECTED)
		lever_end_x = center_x + 40 * cos_angle
		lever_end_y = center_y + 40 * sin_angle
		lever = canvas.create_line(center_x, center_y, lever_end_x, lever_end_y, fill = Shape.COLOR_UNSELECTED)
		lever_end = canvas.create_oval(lever_end_x - 3, lever_end_y - 3, lever_end_x + 3, lever_end_y + 3, fill = Shape.COLOR_UNSELECTED)
		
		return  RotatedRectangle(canvas, [point1, point2, point3, point4, line1, line2, line3, line4, center, lever, lever_end]), point3
		
	def has_subshape(self, subshape):
		for pos in RotatedRectangle._points_pos:
			if self.sub_shapes[pos] == subshape:
				return True
		return False
		
	def to_string(self):
		return super(RotatedRectangle, self).to_string() + ' ' + str(self.angle)
	
	def pull_subshape(self, subshape, x, y):
		# rotate other corners and lines
		super(RotatedRectangle, self).pull_subshape(subshape, x, y)
		
		# rotate
		lever_end = self.sub_shapes[10]
		center = self.sub_shapes[8]
		center_coord = self.canvas.coords(center)
		center_x = center_coord[0] + 3
		center_y = center_coord[1] + 3
		
		if subshape == lever_end:
			d = math.sqrt((x - center_x)**2 + (y - center_y)**2)
			if d==0:
				self.angle=0
			else:
				sin_angle = (y-center_y)/d
				cos_angle = (x-center_x)/d
				self.angle = math.acos(cos_angle)
				if sin_angle<0:
					self.angle = 2 * math.pi - self.angle
			
			# rotate points
			sin_angle = math.sin(self.angle)
			cos_angle = math.cos(self.angle)
			corner1 = self.sub_shapes[0]
			corner2 = self.sub_shapes[1]
			corner3 = self.sub_shapes[2]
			corner4 = self.sub_shapes[3]
			line1 = self.sub_shapes[4]
			line2 = self.sub_shapes[5]
			line3 = self.sub_shapes[6]
			line4 = self.sub_shapes[7]
			
			corner1_coord = self.canvas.coords(corner1)
			corner2_coord = self.canvas.coords(corner2)
			corner4_coord = self.canvas.coords(corner4)
			half_width = math.sqrt((corner1_coord[0] - corner2_coord[0])**2 + (corner1_coord[1] - corner2_coord[1])**2)/2
			half_height = math.sqrt((corner1_coord[0] - corner4_coord[0])**2 + (corner1_coord[1] - corner4_coord[1])**2)/2
			
			new_x1 = center_x - half_width * cos_angle + half_height * sin_angle
			new_y1 = center_y - half_width * sin_angle - half_height * cos_angle
			
			new_x2 = center_x + half_width * cos_angle + half_height * sin_angle
			new_y2 = center_y + half_width * sin_angle - half_height * cos_angle
			
			new_x3 = center_x + half_width * cos_angle - half_height * sin_angle
			new_y3 = center_y + half_width * sin_angle + half_height * cos_angle
			
			new_x4 = center_x - half_width * cos_angle - half_height * sin_angle
			new_y4 = center_y - half_width * sin_angle + half_height * cos_angle
			
			self.canvas.coords(corner1, (new_x1 - 3, new_y1 - 3, new_x1 + 3, new_y1 + 3))
			self.canvas.coords(corner2, (new_x2 - 3, new_y2 - 3, new_x2 + 3, new_y2 + 3))
			self.canvas.coords(corner3, (new_x3 - 3, new_y3 - 3, new_x3 + 3, new_y3 + 3))
			self.canvas.coords(corner4, (new_x4 - 3, new_y4 - 3, new_x4 + 3, new_y4 + 3))
			self.canvas.coords(line1, (new_x1, new_y1, new_x2, new_y2))
			self.canvas.coords(line2, (new_x2, new_y2, new_x3, new_y3))
			self.canvas.coords(line3, (new_x3, new_y3, new_x4, new_y4))
			self.canvas.coords(line4, (new_x4, new_y4, new_x1, new_y1))
			
		# move the lever and lever end later
		lever_end_x = center_x + 40 * math.cos(self.angle)
		lever_end_y = center_y + 40 * math.sin(self.angle)
		lever = self.sub_shapes[9]
		self.canvas.coords(lever, (center_x, center_y, lever_end_x, lever_end_y))
		self.canvas.coords(lever_end, (lever_end_x - 3, lever_end_y - 3, lever_end_x + 3, lever_end_y + 3))		

		