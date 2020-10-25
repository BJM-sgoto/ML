import random
import numpy as np
from collections import deque, defaultdict
import datetime
import tensorflow.compat.v1 as tf

from board import Board, Game
from mcts import MCTSPlayer
from mcts_pure import TreeNode
from neural_net import PolicyValueNet

from constants import BOARD_WIDTH, BOARD_HEIGHT, N_IN_ROW, LEARN_RATE, LR_MULTIPLIER, TEMPERATURE, NUM_PLAYOUT, BUFFER_SIZE, BATCH_SIZE, KL_TARGET, CHECK_FREQ, GAME_BATCH_NUM, NUM_PURE_MCTS_PLAYOUT, PLAY_BATCH_SIZE, EPOCHS, RANDOM_SEED, P_X, P_O, P_E

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.disable_v2_behavior()

TRAIN_MODEL_NAME = 'train_model'
BEST_MODEL_NAME = 'best_model'

TRAIN_MODEL_PATH = './model/train_model'
BEST_MODEL_PATH = './model/best_model'
CACHE_FILE_PATH = './model/cache.txt'

class TrainPipeline:
	def __init__(self, resume=False):
		self.board_width = BOARD_WIDTH
		self.board_height = BOARD_HEIGHT
		self.n_in_row = N_IN_ROW
		self.board = Board(
			width=self.board_width,
			height=self.board_height,
			n_in_row=self.n_in_row)
		self.game = Game(self.board)
		self.learn_rate = LEARN_RATE
		self.lr_multiplier = LR_MULTIPLIER
		self.temp = TEMPERATURE
		self.n_playout = NUM_PLAYOUT
		self.buffer_size = BUFFER_SIZE
		self.batch_size = BATCH_SIZE
		self.data_buffer = deque(maxlen=self.buffer_size)
		self.play_batch_size = PLAY_BATCH_SIZE
		self.epochs = EPOCHS
		self.kl_targ = KL_TARGET
		self.check_freg = CHECK_FREQ
		self.game_batch_num = GAME_BATCH_NUM
		self.pure_mcts_playout_num = NUM_PURE_MCTS_PLAYOUT
		self.resume = resume
	
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = False
		config.gpu_options.per_process_gpu_memory_fraction = 0.45
		self.session = tf.Session(config=config)
		if resume:
			self.train_net = PolicyValueNet(self.session, TRAIN_MODEL_NAME, self.board_width, self.board_height, model_file=TRAIN_MODEL_PATH)
			self.best_net = PolicyValueNet(self.session, BEST_MODEL_NAME, self.board_width, self.board_height, model_file=BEST_MODEL_PATH)
		else:
			self.train_net = PolicyValueNet(self.session, TRAIN_MODEL_NAME, self.board_width, self.board_height)
			self.best_net = PolicyValueNet(self.session, BEST_MODEL_NAME, self.board_width, self.board_height)
			self.session.run(tf.global_variables_initializer())
		self._copy_model_op = self.make_copy_model_op(source_model=self.train_net, target_model=self.best_net)
		self.mcts_player = MCTSPlayer(
			self.train_net.policy_value_fn,
			n_playout = self.n_playout,
			is_selfplay=True)
		'''	
		for var in tf.trainable_variables():
			print(var)
		print('------------------')
		for var in tf.global_variables():
			print(var)
		'''
	def make_copy_model_op(self, source_model, target_model):
		source_model_variables = tf.trainable_variables(scope=source_model.model_name)
		target_model_variables = tf.trainable_variables(scope=target_model.model_name)
		copy_ops = []
		for i in range(len(source_model_variables)):
			copy_ops.append(tf.assign(target_model_variables[i], source_model_variables[i]))
		return copy_ops
	
	def copy_model(self):
		self.session.run(self._copy_model_op)
			
	# agument data
	def get_equi_data(self, play_data):
		extend_data = []
		for state, mcts_prob, winner in play_data:
			for i in [1,2,3,4]:
				pi_board = mcts_prob.reshape(self.board_height, self.board_width)
				# rotate
				equi_state = np.rot90(state, i)
				equi_mcts_prob = np.rot90(pi_board, i)
				extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
				# flip
				equi_state = np.fliplr(equi_state)
				equi_mcts_prob = np.fliplr(equi_mcts_prob)
				extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
				
		return extend_data
		
	def collect_selfplay_data(self, n_games=1):
		for i in range(n_games):
			winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
			play_data = list(play_data)[:]
			self.episode_len = len(play_data)
			play_data = self.get_equi_data(play_data)
			self.data_buffer.extend(play_data)
		
	
	def policy_update(self):
		mini_batch = random.sample(self.data_buffer, self.batch_size)
		state_batch = np.float32([data[0] for data in mini_batch])
		mcts_probs_batch = np.float32([data[1] for data in mini_batch])
		winner_batch = np.float32([data[2] for data in mini_batch])
		winner_batch = np.reshape(winner_batch, (-1, 1))
		old_probs, old_v = self.train_net.policy_value_fn(state_batch)
		for i in range(self.epochs):
			loss, entropy = self.train_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
			new_probs, new_v = self.train_net.policy_value_fn(state_batch)
			
			kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
			if kl>self.kl_targ * 4:
				break
		if kl>self.kl_targ*2 and self.lr_multiplier>0.1:
			self.lr_multiplier/=1.5
		elif kl<self.kl_targ/2 and self.lr_multiplier<10:
			self.lr_multiplier*=1.5
		print('kl:{:5f}, lr_multiplier:{:2f}, loss:{:5f}, entropy:{:3f}'.format(kl, self.lr_multiplier, loss, entropy))
		return loss, entropy
	
	def policy_evaluate(self, n_games=10):
		train_player = MCTSPlayer(
			self.train_net.policy_value_fn,
			n_playout = self.n_playout,
			is_selfplay=False)
		best_player =MCTSPlayer(
			self.best_net.policy_value_fn,
			n_playout = self.n_playout,
			is_selfplay=False)
		win_count = defaultdict(int)
		for i in range(n_games):
			winner = self.game.start_play(
				train_player,
				best_player,
				start_player=i%2,
				is_shown=0)
			win_count[winner]+=1
		win_ratio = 1.0 *(win_count[P_X] + 0.5 * win_count[P_E]) / n_games
		print('win: {}, loss: {}, tie: {}'.format(win_count[P_X], win_count[P_O], win_count[P_E]))
		return win_ratio
		
	def run(self):
		try:
			for i in range(self.game_batch_num):
				start = datetime.datetime.now()
				self.collect_selfplay_data(self.play_batch_size)
				end = datetime.datetime.now()
				print('--------------\nTime to make data', end - start)
				print('batch i:{}, episode_len:{}'.format(i+1, self.episode_len))
				start = datetime.datetime.now()
				if len(self.data_buffer)>self.batch_size:
					loss, entropy = self.policy_update()
				end = datetime.datetime.now()
				print('Time to update', end - start)
				if (i+1)%self.check_freg==0:
					print('current self-play batch: {}'.format(i+1))
					win_ratio = self.policy_evaluate()
					self.train_net.save_model(TRAIN_MODEL_PATH)
					if win_ratio>0.55:
						print('New best policy!!!')
						self.copy_model()
						self.best_net.save_model(BEST_MODEL_PATH)
		except KeyboardInterrupt:
			print('\n\rQuit')
			
if __name__ == '__main__':
	training_pipleline = TrainPipeline(resume=True)
	training_pipleline.run()