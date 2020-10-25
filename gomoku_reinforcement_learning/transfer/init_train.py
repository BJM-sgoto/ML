import random
import numpy as np
from neural_net import PolicyValueNet
from constants import BOARD_SIZE, P_O, P_X, P_E, N_IN_ROW, TRAIN_MODEL_NAME, TRAIN_MODEL_PATH, RANDOM_SEED
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED+1)

BATCH_SIZE = 512


class SimpleBoard:
	def __init__(self):
		self.reset()
		
	def reset(self):
		self._state = np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.int8) * P_E
		self.current_player = np.random.choice([P_X, P_O])
		self._moves = []
		self.availables = list(range(BOARD_SIZE * BOARD_SIZE))
	
	def get_state(self):
		return self._state * self.current_player
		
	def get_result(self):
		if len(self._moves)<2*N_IN_ROW:
			return False, P_E
			
		move = self._moves[-1]
		y = move//BOARD_SIZE
		x = move%BOARD_SIZE
		player = self._state[y, x]
		
		# check horizontal cells
		score = 1
		min_x = max(-1, x - N_IN_ROW)
		for i in range(x-1, min_x, -1):
			if self._state[y, i] == player:
				score += 1
			else:
				break
		max_x = min(BOARD_SIZE, x + N_IN_ROW)
		for i in range(x + 1, max_x):
			if self._state[y, i] == player:
				score += 1
			else:
				break
			
		if score >=	N_IN_ROW:
			return True, player
		
		# check vertical cells
		score = 1
		min_y = max(-1, y - N_IN_ROW)
		for i in range(y - 1, min_y, -1):
			if self._state[i, x] == player:
				score += 1
			else:
				break
		max_y = min(BOARD_SIZE, y + N_IN_ROW)
		for i in range(y + 1, max_y):
			if self._state[i, x] == player:
				score += 1
			else:
				break
		
		if score>=N_IN_ROW:
			return True, player
			
		# check NW->ES diagonal cells
		score = 1
		d = min(x - min_x, y - min_y)
		for i in range(-1, -d, -1):
			if self._state[y+i, x+i]==player:
				score += 1
			else:
				break
			
		d = min(max_x - x, max_y - y)
		for i in range(1, d):
			if self._state[y+i, x+i]==player:
				score += 1
			else:
				break
				
		if score >= N_IN_ROW:
			return True, player
			
		# check NE->WS diagonal cells
		score = 1
		d = min(max_x - x, y - min_y)
		for i in range(1, d):
			if self._state[y-i, x+i]==player:
				score += 1
			else:
				break
				
		d = min(x - min_x, max_y - y)
		for i in range(1, d):
			if self._state[y+i, x-i]==player:
				score += 1
			else:
				break
		
		if score >= N_IN_ROW:
			return True, player
		
		if len(self.availables)<=0:
			return True, P_E
		
		return False, P_E
	
	def step(self, move):
		y = move//BOARD_SIZE
		x = move%BOARD_SIZE
		self._state[y, x] = self.current_player
		self._moves.append(move)
		self.availables.remove(move)
		self.current_player = P_O + P_X - self.current_player
	
	def undo(self):
		move = self._moves.pop()
		y = move//BOARD_SIZE
		x = move%BOARD_SIZE
		self._state[y, x] = P_E
		self.availables.append(move)
		self.current_player = P_O + P_X - self.current_player

class InitModel:
	def __init__(self, resume=False):
		self.env = SimpleBoard()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = False
		config.gpu_options.per_process_gpu_memory_fraction = 0.45
		self.session = tf.Session(config=config)
		if resume:
			self.net = PolicyValueNet(self.session, TRAIN_MODEL_NAME, BOARD_SIZE, BOARD_SIZE, model_file=TRAIN_MODEL_PATH)
		else:
			self.net = PolicyValueNet(self.session, TRAIN_MODEL_NAME, BOARD_SIZE, BOARD_SIZE)
			self.session.run(tf.global_variables_initializer())

	def make_dataset(self, datafile='./data.txt'):
		dataset = []
		f = open(datafile, 'r')
		s = f.readline().strip()
		while s!=None and len(s)>0:
			episode = np.int32(eval(s))
			#episode = np.int32([episode//BOARD_SIZE, episode%BOARD_SIZE])
			dataset.append(episode)
			s = f.readline().strip()
		f.close()
		return dataset
	
	def extend_dataset(self, dataset):
		extended_dataset = []
		for move in dataset: 
			move_y = move // BOARD_SIZE
			move_x = move % BOARD_SIZE
			# rotate 0 degrees clockwise
			# move_x_0 = move_x
			# move_y_0 = move_y
			
			#rotate 90 degrees clockwise
			move_x_90 = BOARD_SIZE - 1 - move_y
			move_y_90 = move_x
			
			# rotate 180 degrees clockwise
			move_x_180 = BOARD_SIZE - 1 - move_y_90
			move_y_180 = move_x_90
			
			# rotate 270 degrees clockwise
			move_x_270 = BOARD_SIZE - 1 - move_y_180
			move_y_270 = move_x_180
			
			extended_dataset.append(move_y * BOARD_SIZE + move_x)
			extended_dataset.append(move_y_90 * BOARD_SIZE + move_x_90)
			extended_dataset.append(move_y_180 * BOARD_SIZE + move_x_180)
			extended_dataset.append(move_y_270 * BOARD_SIZE + move_x_270)
			
			# exchange move_x and move_y 
			extended_dataset.append(move_x * BOARD_SIZE + move_y)
			extended_dataset.append(move_x_90 * BOARD_SIZE + move_y_90)
			extended_dataset.append(move_x_180 * BOARD_SIZE + move_y_180)
			extended_dataset.append(move_x_270 * BOARD_SIZE + move_y_270)
		return extended_dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def train_on_episode(self, episode):
		current_states = []
		current_availables = []
		current_availables_len = []
		next_states = []
		self.env.reset()
		move_counts = []
		next_ends_s = []
		winners_s = []
		
		# ------------- gather current states and next states ----------- #
		for move in episode[:-1]:
			current_states.append(self.env.get_state())
			current_availables.append(self.env.availables.copy())
			current_availables_len.append(len(self.env.availables))
			for t_move in self.env.availables:
				self.env.step(t_move)
				next_states.append(self.env.get_state())
				
				end, winner = self.env.get_result()
				next_ends_s.append(end)
				winners_s.append(winner)
				
				self.env.undo()
			self.env.step(move)
		current_states = np.int8(current_states)
		next_states = np.int8(next_states)
		
		# ------------- divide data to batches to compute ----------- #
		current_policies, current_values = self.net.policy_value_fn(current_states)
		print('last current state', current_states[-1])
		print('current_values', current_values)
		exit()
		next_values_s = []
		for i in range(0, next_states.shape[0], BATCH_SIZE):
			end_i = min(i+BATCH_SIZE, next_states.shape[0])
			_, batch_next_values_s = self.net.policy_value_fn(next_states[i: end_i])
			next_values_s.append(batch_next_values_s)
		next_values_s = np.concatenate(next_values_s, axis=0)
		#print('last current state', current_states[-1])
		
		# ------------- make target evaluation and policy ----------- #
		n = current_policies.shape[0]
		target_values = np.zeros([n, 1], dtype=np.float32)
		target_policies = np.zeros_like(current_policies)
		for i in range(n):
			start_id = sum(current_availables_len[:i])
			end_id = sum(current_availables_len[:i+1])
			
			next_values = next_values_s[start_id: end_id]
			next_ends = next_ends_s[start_id: end_id]
			availables = current_availables[i]
			# check teriminal states
			can_terminate = False
			for j, next_end in enumerate(next_ends):
				if next_end:
					if winner==P_E:
						target_values[i] = 0
					else:
						target_values[i] = 1
					action = availables[j]
					target_policies[i,availables[j]] = 1.0
					can_terminate = True			
			if not can_terminate:
				next_values = -next_values
				#print('Current policy', current_policies[i][availables])
				#print('Next value', next_values[:, 0])
				
				target_values[i, 0] =  np.sum(next_values[:, 0] * current_policies[i][availables])
				action = availables[np.argmax(next_values)]
				target_policies[i, action] = 1.0
			else:
				target_policies[i] = target_policies[i]/ np.sum(target_policies[i])
			
		print('target_values', target_values)
		print('target_actions', np.argmax(target_policies, axis=1))
			
		# ------------- train ----------- #
		loss_val, _ = self.net.train_step(current_states, target_policies, target_values, 1e-3)
		return loss_val
		
	def train(self, num_epoch=10, datafile='./data.txt'):
		dataset = self.make_dataset(datafile)
		dataset = self.extend_dataset(dataset)
		count_to_save = 0
		n_episode = len(dataset)
		for i in range(num_epoch):
			self.shuffle_dataset(dataset)
			for j in range(0, n_episode):
				loss_val = self.train_on_episode(dataset[j])
				count_to_save += 1
				print('Epoc {:02d}, Ep {:06d}, Count {:03d}, Loss {:06f}'.format(i, j, count_to_save, loss_val))
				if count_to_save>=500:
					count_to_save = 0
					self.net.save_model(TRAIN_MODEL_PATH)
		self.session.close()
model = InitModel(resume=False)
model.train(num_epoch=10, datafile='./data.txt')