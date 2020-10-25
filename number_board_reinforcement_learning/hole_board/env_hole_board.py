# 4 actions
# (height*width)! state
# 1 solved state => sparse environment

# TASK change the state of whole board
# TASK change the state of one piece
# TASK be able customize output

import numpy as np

REWARD_NOT_SOLVED 	= -0.10
REWARD_UNCHANGED 		=  0.00
REWARD_SOLVED 			=  1.00

class Environment:
	LEFT = 0
	UP = 1
	RIGHT = 2 # opposite of left
	DOWN = 3 # opposite of up
	
	def __init__(self, height, width):
		self.height = height
		self.width = width
		#self.state_size = [height * width, int(np.ceil(np.log2(height * width)))]
		self.state_size = [height, width]
		
		_state = np.arange(height * width) # hole_pos = [0, 0]
		self.hole_pos = [0, 0]
		_state = _state.reshape(height, width)
		self._state = _state
		self._solved_state = _state.copy()
				
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
		return self._state.copy()
	
	def sample_action(self):
		return np.random.choice(self.get_valid_actions())
		
	def get_result(self,):
		for i in range(self.height):
			for j in range(self.width):
				if self._state[i,j] != self._solved_state[i,j]:
					return REWARD_NOT_SOLVED
		return REWARD_SOLVED
	
	# this function returns reward
	def step(self, action):
		hole_y, hole_x = self.hole_pos
		if action==Environment.UP and hole_y>0:
			temp = self._state[hole_y-1, hole_x]
			self._state[hole_y-1, hole_x] = self._state[hole_y, hole_x]
			self._state[hole_y, hole_x] = temp
			self.hole_pos[0] -= 1
		elif action==Environment.DOWN and hole_y<self.height-1:
			temp = self._state[hole_y+1, hole_x]
			self._state[hole_y+1, hole_x] = self._state[hole_y, hole_x]
			self._state[hole_y, hole_x] = temp
			self.hole_pos[0] += 1
		elif action==Environment.LEFT and hole_x>0:
			temp = self._state[hole_y, hole_x-1]
			self._state[hole_y, hole_x-1] = self._state[hole_y, hole_x]
			self._state[hole_y, hole_x] = temp
			self.hole_pos[1] -= 1
		elif action==Environment.RIGHT and hole_x<self.width-1:
			temp = self._state[hole_y, hole_x+1]
			self._state[hole_y, hole_x+1] = self._state[hole_y, hole_x]
			self._state[hole_y, hole_x] = temp
			self.hole_pos[1] += 1
		else: 
			return REWARD_UNCHANGED
		return self.get_result()
		
	def invert_action(self, action):
		if action>=2:
			action -= 2
		else:
			action += 2
		return action
	
	def undo(self, action):
		return self.step(self.invert_action(action))
	
	def reset(self, reset_to_solved_state=True, n_step=20):
		self._state = self._solved_state.copy()
		self.hole_pos[1] = 0
		self.hole_pos[0] = 0
		if reset_to_solved_state==False:
			for i in range(n_step):
				action = self.sample_action()
				self.step(action)
		
	def export_state(self):
		return [self._state.copy(), self.hole_pos.copy()]
	
	def import_state(self, state):
		self._state = state[0]
		self.hole_pos = state[1]

	def get_valid_actions(self):
		valid_actions = [Environment.LEFT, Environment.UP, Environment.RIGHT, Environment.DOWN]
		hole_y, hole_x = self.hole_pos
		if hole_y<=0:
			valid_actions.remove(Environment.UP)
		elif hole_y>=self.height-1:
			valid_actions.remove(Environment.DOWN)
		
		if hole_x<=0:
			valid_actions.remove(Environment.LEFT)
		elif hole_x>=self.width-1:
			valid_actions.remove(Environment.RIGHT)
			
		return valid_actions
	
	def get_valid_action_mask(self):
		valid_actions = self.get_valid_actions()
		mask = np.zeros(self.n_actions, dtype=np.float32)
		for valid_action in valid_actions:
			mask[valid_action] = 1.0
		return mask
		
	def export_state(self):
		return (
			self._state.copy(),
			self.hole_pos.copy())
			
	def import_state(self, state):
		self._state = state[0].copy()
		self.hole_pos = state[1].copy()
		
'''
env = Environment(4,4)
env.reset()
actions = []
inverted_actions = []
for i in range(20):
	action = env.sample_action()
	env.step(action)
	actions.append(action)
	inverted_actions.insert(0, env.invert_action(action))
	env.render()

print(actions, len(actions))
env.render()

for action in inverted_actions:
	env.step(action)
env.render()
'''