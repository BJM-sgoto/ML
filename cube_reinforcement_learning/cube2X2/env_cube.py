# This environment can be apply for cube 2X2 or 3X3

import numpy as np

CUBE_SIZE = 2
REWARD_NOT_SOLVED = -0.10
REWARD_SOLVED = 1.00

class Environment:
	# action
	ROTATE_FRONT = 0
	ROTATE_RIGHT = 1
	ROTATE_UP = 2
	ROTATE_LEFT = 3
	ROTATE_BACK = 4
	ROTATE_DOWN = 5
	
	ROTATE_REVERSE_FRONT = 6
	ROTATE_REVERSE_RIGHT = 7
	ROTATE_REVERSE_UP = 8
	ROTATE_REVERSE_LEFT = 9
	ROTATE_REVERSE_BACK = 10
	ROTATE_REVERSE_DOWN = 11
	
	def __init__(self):
		self._up = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 1
		self._left = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 2
		self._back = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 3
		self._right = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 4
		self._front = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 5
		self._down = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 6
		self.n_action = 12 # only rotate side clockwise, when display, display inside faces-> rotate anti_clockwise
		self.action_space = list(range(self.n_action))
		self.state_size = [CUBE_SIZE * CUBE_SIZE * 6, 6]
		self._lookup_table = self._make_lookup_table() # convert numbers to binary format
	
	def reset(self):
		self._up = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 1
		self._left = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 2
		self._back = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 3
		self._right = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 4
		self._front = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 5
		self._down = np.ones([CUBE_SIZE,CUBE_SIZE], dtype=np.int8) * 6
		
	def _to_string(self):
		sides = [self._up, self._left, self._back, self._right, self._front, self._down]
		s = ''
		space = '   ' * CUBE_SIZE
		for i in range(CUBE_SIZE):
			s += space
			for j in range(CUBE_SIZE):
				s+='[' + str(sides[0][i,j]) + ']'
			s+='\n'
		for m in range(CUBE_SIZE):
			for i in range(1,5):
				for n in range(CUBE_SIZE):
					s+='[' + str(sides[i][m,n]) + ']'
			s+='\n'
		
		for i in range(CUBE_SIZE):
			s += space
			for j in range(CUBE_SIZE):
				s+='[' + str(sides[5][i,j]) + ']'
			if i!=CUBE_SIZE-1:
				s+='\n'
		return s
	
	def _make_lookup_table(self):
		return np.float32(np.eye(6))
	
	def render(self):
		print(self._to_string())
		
	def sample_action(self):
		return np.random.choice(self.n_action)
	
	def get_result(self):
		for i in range(CUBE_SIZE):
			for j in range(CUBE_SIZE):	
				if self._up[i,j]!=1 or self._left[i,j]!=2 or self._back[i,j]!=3 or self._right[i,j]!=4 or self._front[i,j]!=5 or self._down[i,j]!=6:
					return REWARD_NOT_SOLVED
		return REWARD_SOLVED
	
	def get_state(self):
		flattened_sides = np.concatenate([
			self._up.flatten(),
			self._left.flatten(),
			self._back.flatten(),
			self._right.flatten(),
			self._front.flatten(),
			self._down.flatten()])
		return np.take(self._lookup_table, flattened_sides-1, axis=0)
	
	# return 1 -> end
	# return 0 -> not end
	def step(self, action):
		if action==Environment.ROTATE_FRONT:
			self._front = np.rot90(self._front, 1)
			temp = self._left[:, 0].copy()
			self._left[:, 0] = self._down[CUBE_SIZE-1]
			self._down[CUBE_SIZE-1] = self._right[:, CUBE_SIZE-1][::-1]
			self._right[:, CUBE_SIZE-1] = self._up[0]
			self._up[0] = temp[::-1]
			
		elif action==Environment.ROTATE_REVERSE_FRONT:
			self._front = np.rot90(self._front, 3)
			temp = self._left[:, 0].copy()
			self._left[:, 0] = self._up[0][::-1]
			self._up[0] = self._right[:,CUBE_SIZE-1]
			self._right[:,CUBE_SIZE-1] = self._down[CUBE_SIZE-1][::-1]
			self._down[CUBE_SIZE-1] = temp
			
		elif action==Environment.ROTATE_RIGHT:
			self._right = np.rot90(self._right, 1)
			temp = self._front[:, 0].copy()
			self._front[:, 0] = self._down[:,CUBE_SIZE-1][::-1]
			self._down[:,CUBE_SIZE-1] = self._back[:,CUBE_SIZE-1]
			self._back[:,CUBE_SIZE-1] = self._up[:,CUBE_SIZE-1]
			self._up[:,CUBE_SIZE-1] = temp[::-1]
			
		elif action==Environment.ROTATE_REVERSE_RIGHT:
			self._right = np.rot90(self._right, 3)
			temp = self._front[:, 0].copy()
			self._front[:,0] = self._up[:,CUBE_SIZE-1][::-1]
			self._up[:,CUBE_SIZE-1] = self._back[:,CUBE_SIZE-1]
			self._back[:,CUBE_SIZE-1] = self._down[:,CUBE_SIZE-1]
			self._down[:,CUBE_SIZE-1] = temp[::-1]
		
		elif action==Environment.ROTATE_UP:
			self._up = np.rot90(self._up, 1)
			temp = self._front[0].copy()
			self._front[0] = self._right[0]
			self._right[0] = self._back[0]
			self._back[0] = self._left[0]
			self._left[0] = temp
			
		elif action==Environment.ROTATE_REVERSE_UP:
			self._up = np.rot90(self._up, 3)
			temp = self._front[0].copy()
			self._front[0] = self._left[0]
			self._left[0] = self._back[0]
			self._back[0] = self._right[0]
			self._right[0] = temp
			
		elif action==Environment.ROTATE_LEFT:
			self._left = np.rot90(self._left, 1)
			temp = self._front[:,CUBE_SIZE-1].copy()
			self._front[:,CUBE_SIZE-1] = self._up[:,0][::-1]
			self._up[:,0] = self._back[:,0]
			self._back[:,0] = self._down[:,0]
			self._down[:,0] = temp[::-1]
			
		elif action==Environment.ROTATE_REVERSE_LEFT:
			self._left = np.rot90(self._left, 3)
			temp = self._front[:,CUBE_SIZE-1].copy()
			self._front[:,CUBE_SIZE-1] = self._down[:,0][::-1]
			self._down[:,0] = self._back[:,0]
			self._back[:,0] = self._up[:,0]
			self._up[:,0] = temp[::-1]
			
		elif action==Environment.ROTATE_BACK:
			self._back = np.rot90(self._back, 1)
			temp = self._left[:,CUBE_SIZE-1].copy()
			self._left[:,CUBE_SIZE-1] = self._up[CUBE_SIZE-1][::-1]
			self._up[CUBE_SIZE-1] = self._right[:,0]
			self._right[:,0] = self._down[0][::-1]
			self._down[0] = temp
			
		elif action==Environment.ROTATE_REVERSE_BACK:
			self._back = np.rot90(self._back, 3)
			temp = self._left[:,CUBE_SIZE-1].copy()
			self._left[:,CUBE_SIZE-1] = self._down[0]
			self._down[0] = self._right[:,0][::-1]
			self._right[:,0] = self._up[CUBE_SIZE-1]
			self._up[CUBE_SIZE-1] = temp[::-1]
		
		elif action==Environment.ROTATE_DOWN:
			self._down = np.rot90(self._down, 1)
			temp = self._front[CUBE_SIZE-1].copy()
			self._front[CUBE_SIZE-1] = self._left[CUBE_SIZE-1]
			self._left[CUBE_SIZE-1] = self._back[CUBE_SIZE-1]
			self._back[CUBE_SIZE-1] = self._right[CUBE_SIZE-1]
			self._right[CUBE_SIZE-1] = temp
			
		elif action==Environment.ROTATE_REVERSE_DOWN:
			self._down = np.rot90(self._down, 3)
			temp = self._front[CUBE_SIZE-1].copy()
			self._front[CUBE_SIZE-1] = self._right[CUBE_SIZE-1]
			self._right[CUBE_SIZE-1] = self._back[CUBE_SIZE-1]
			self._back[CUBE_SIZE-1] = self._left[CUBE_SIZE-1]
			self._left[CUBE_SIZE-1] = temp
			
		return self.get_result()			
	
	def undo(self, action):
		if action>=6:
			action-=6
		else:
			action+=6
		return self.step(action)	

	def get_sides(self):
		return [
			self._up.copy(),
			self._left.copy(),
			self._back.copy(),
			self._right.copy(),
			self._front.copy(),
			self._down.copy()]
	
	def init_from_sides(self, sides):
		self._up = sides[0]
		self._left = sides[1]
		self._back = sides[2]
		self._right = sides[3]
		self._front = sides[4]
		self._down = sides[5]

'''
env = Environment()
env.reset()
actions = []
for i in range(10):
	action = env.sample_action()
	env.step(action)
	
	if action>=6:	
		action-=6
	else:
		action+=6
	actions.insert(0, action)

sides = env.get_sides()
env.reset()
env.render()
env.init_from_sides(sides)
env.render()
for action in actions:
	env.step(action)
	env.render()
'''