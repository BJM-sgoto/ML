# (height*width)! state
# (height-1)*(width-1) actions => many actions
# 1 solved state => sparse environment

import numpy as np
# spinner size in this environment is 2X2

REWARD_NOT_SOLVED 	= -0.10
REWARD_SOLVED 			=  1.00

class Environment:
	def __init__(self, height, width):
		self.height = height
		self.width = width
		self.state_size = [height * width, int(np.log(height * width) + 1)]
		self.n_actions = (height-1) * (width-1) * 2 # clockwise and anti-clockwise
		self.actions = np.arange(self.n_actions)
		
		_state = np.arange(height * width)
		_state = _state.reshape(height, width)
		self._state = _state
		self._solved_state = _state.copy()
		self._lookup_table = self._make_lookup_table(height, width) # convert numbers to binary format
		
	def _make_lookup_table(self, height, width):
		num_pieces = self.state_size[0]
		piece_size = self.state_size[1]
		lookup_table = np.zeros([num_pieces, piece_size], dtype=np.float32)
		piece_format = '{0:0' + str(piece_size) + 'b}'
		for i in range(num_pieces):
			piece_code = piece_format.format(i)
			for j in range(piece_size):
				if piece_code[j] == '1':
					lookup_table[i,j] = 1.0
		return lookup_table
					
	def _to_string(self):
		s = ''
		for i in range(self.height):
			for j in range(self.width):
				s += '[{:02d}]'.format(self._state[i,j])
			s += '\n'
		return s
		
	def render(self):
		print(self._to_string())
	
	def get_state(self):
		return np.take(self._lookup_table, self._state.flatten(), axis=0)
	
	def get_result(self):
		for i in range(self.height):
			for j in range(self.width):
				if self._state[i,j] != self._solved_state[i,j]:
					return REWARD_NOT_SOLVED
		return REWARD_SOLVED
	
	def sample_action(self):
		return np.random.choice(self.actions)
	
	def invert_action(self, action):
		half_n_actions = self.n_actions//2
		if action>=half_n_actions:
			action -= half_n_actions
		else:
			action += half_n_actions
		return action
	
	# this function returns reward
	# reward==0 -> not end
	# reward==1 -> end
	def step(self, action):
		clockwise = action//((self.width-1)*(self.width-1))
		spin_pos = action%((self.width-1)*(self.width-1))
		spin_y, spin_x = spin_pos//(self.width-1), spin_pos%(self.width-1)
		if clockwise>0:
			temp = self._state[spin_y, spin_x]
			self._state[spin_y, spin_x] = self._state[spin_y+1, spin_x]
			self._state[spin_y+1, spin_x] = self._state[spin_y+1, spin_x+1]
			self._state[spin_y+1, spin_x+1] = self._state[spin_y, spin_x+1]
			self._state[spin_y, spin_x+1] = temp
		else:
			temp = self._state[spin_y, spin_x]
			self._state[spin_y, spin_x] = self._state[spin_y, spin_x+1]
			self._state[spin_y, spin_x+1] = self._state[spin_y+1, spin_x+1]
			self._state[spin_y+1, spin_x+1] = self._state[spin_y+1, spin_x]
			self._state[spin_y+1, spin_x] = temp
		
		return self.get_result()
	
	def undo(self, action):
		self.step(self.invert_action(action))
	
	def reset(self, reset_to_solved_state=True, n_step=10):
		self._state = self._solved_state.copy()
		if reset_to_solved_state==False:
			for i in range(n_step):
				action = self.sample_action()
				self.step(action)

	def export_state(self):
		return self._state.copy()
	
	def import_state(self, state):
		self._state = state.copy()

'''
env = Environment(height=4, width=4)
actions = []
env.render()
for i in range(10):
	action = env.sample_action()
	actions.insert(0, action)
	env.step(action)
env.render()

for action in actions:
	print('Action',action)
	env.undo(action)

env.render()
'''
