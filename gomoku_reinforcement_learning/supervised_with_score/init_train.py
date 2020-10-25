import random
import numpy as np
from board import Board
from neural_net import PolicyValueNet
from constants import BOARD_SIZE, P_O, P_X, P_E, N_IN_ROW, TRAIN_MODEL_NAME, TRAIN_MODEL_PATH, RANDOM_SEED
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

BATCH_SIZE = 40

class SimpleBoard:
	def __init__(self):
		self._moves = []
		self.init_board()
		
	def init_board(self, start_player=P_X):
		self._state = np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.int8) * P_E
		self.current_player = start_player
		self._moves.clear()
		
	def current_state(self):
		return self._state * self.current_player
		
	def get_result(self):
		if len(self._moves)<2*N_IN_ROW:
			return False, P_E
			
		y, x = self._moves[-1]
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
		
		if len(self._moves)>=BOARD_SIZE*BOARD_SIZE:
			return True, P_E
		
		return False, P_E

	def do_move(self, move):
		y, x = move
		self._state[y, x] = self.current_player
		self._moves.append(move)
		self.current_player = P_O + P_X - self.current_player

class InitModel:
	def __init__(self, resume=False):
		self.env = SimpleBoard()
		self.session = tf.Session()
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
			episode = np.int32([episode//BOARD_SIZE, episode%BOARD_SIZE])
			dataset.append(episode)
			s = f.readline().strip()
		f.close()
		return dataset
	
	def extend_dataset(self, dataset):
		extended_dataset = []
		for move in dataset:
			move_y, move_x = move
			move_x = np.int32(move_x)
			move_y = np.int32(move_y)
			move = move_y * BOARD_SIZE + move_x
			if len(move)!=len(set(move)):
				continue
			
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
			
			extended_dataset.append([move_y, move_x])
			extended_dataset.append([move_y_90, move_x_90])
			extended_dataset.append([move_y_180, move_x_180])
			extended_dataset.append([move_y_270, move_x_270])
			
			# exchange move_x and move_y 
			extended_dataset.append([move_x, move_y])
			extended_dataset.append([move_x_90, move_y_90])
			extended_dataset.append([move_x_180, move_y_180])
			extended_dataset.append([move_x_270, move_y_270])
		return extended_dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def train_on_episodes(self, episodes):
		current_states = []
		target_values = []
		target_policies = []
		
		for episode in episodes:
			self.env.init_board(start_player=np.random.choice([P_X, P_O]))
			target_episode_values = []
			target_episode_policies = []
			len_episode = len(episode[0])
			for i in range(len_episode):
				move_y = episode[0][i]
				move_x = episode[1][i]
				current_states.append(self.env.current_state())
				# make target values
				if i%2==0:
					target_episode_values.append(1.0)
				else:
					target_episode_values.append(-1.0)
				
				# make target policies
				target_episode_policy = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.float32)
				target_episode_policy[move_y, move_x] = 1.0
				target_policies.append(target_episode_policy)
				
				# move
				self.env.do_move([move_y, move_x])
			
			# reconfig target values
			target_episode_values = np.float32(target_episode_values)
			_, winner = self.env.get_result()
			if winner==P_E:
				target_episode_values = target_episode_values * 0
			else: # the last action must lead to win
				if len(episode)%2==0:
					target_episode_values = target_episode_values * -1.0
			
			# 
			target_values.append(target_episode_values)

		current_states = np.int8(current_states)
		target_values = np.concatenate(target_values, axis=0)
		target_values = np.expand_dims(target_values, axis=-1)
		target_policies = np.float32(target_policies)
		target_policies = np.reshape(target_policies, [-1, BOARD_SIZE * BOARD_SIZE])
		
		value_loss, policy_loss, _ = self.net.train_step(current_states, target_policies, target_values, 1e-3)
		return value_loss, policy_loss
		
	def train(self, num_epoch=10, datafile='./data.txt'):
		dataset = self.make_dataset(datafile)
		dataset = self.extend_dataset(dataset)
		count_to_save = 0
		n_episode = len(dataset)
		for i in range(num_epoch):
			self.shuffle_dataset(dataset)
			for j in range(0, n_episode, BATCH_SIZE):
				end_j = min(n_episode, j + BATCH_SIZE)
				episodes = dataset[j: end_j]
				value_loss, policy_loss = self.train_on_episodes(episodes)
				print('Epoc {:03d}, Ep {:06d}/ {:06d}, Value Loss {:06f}, Policy Loss {:06f}'.format(i, j, n_episode, value_loss, policy_loss))
				count_to_save += 1
				
				if count_to_save>=500:
					count_to_save = 0
					self.net.save_model(TRAIN_MODEL_PATH)
					
		self.session.close()
		
model = InitModel(resume=False)
model.train(num_epoch=10, datafile='./data.txt')