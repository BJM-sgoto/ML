import random
import numpy as np
from collections import defaultdict
import datetime

from board import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts import MCTSPlayer
from neural_net import PolicyValueNet

from constants import BOARD_WIDTH, BOARD_HEIGHT, N_IN_ROW, LEARN_RATE, LR_MULTIPLIER, TEMPERATURE, NUM_PLAYOUT, C_PUCT, BUFFER_SIZE, BATCH_SIZE, KL_TARGET, CHECK_FREQ, GAME_BATCH_NUM, BEST_WIN_RATIO, NUM_PURE_MCTS_PLAYOUT, PLAY_BATCH_SIZE, EPOCHS, RANDOM_SEED, P_X, P_O, P_E

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CURRENT_MODEL_PATH = './model/current_model'
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
		self.c_puct = C_PUCT
		self.buffer_size = BUFFER_SIZE
		self.batch_size = BATCH_SIZE
		#self.data_buffer = deque(maxlen=self.buffer_size)
		self.play_batch_size = PLAY_BATCH_SIZE
		self.epochs = EPOCHS
		self.kl_targ = KL_TARGET
		self.check_freg = CHECK_FREQ
		self.game_batch_num = GAME_BATCH_NUM
		self.best_win_ratio = BEST_WIN_RATIO
		self.pure_mcts_playout_num = NUM_PURE_MCTS_PLAYOUT
		self.resume = resume
		
		
		if resume:
			self.read_cache(CACHE_FILE_PATH)
			self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=CURRENT_MODEL_PATH)
		else:
			self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
		self.mcts_player = MCTSPlayer(
			self.policy_value_net.policy_value_fn,
			c_puct = self.c_puct,
			n_playout = self.n_playout,
			is_selfplay=True)
			
	def write_cache(self, cache_file):
		f = open(cache_file, 'w')
		f.write(str(self.learn_rate) + '\n')
		f.write(str(self.lr_multiplier) + '\n')
		f.write(str(self.best_win_ratio) + '\n')
		f.write(str(self.pure_mcts_playout_num) + '\n')
		f.close()
		
	def read_cache(self, cache_file):
		f = open(cache_file, 'r')
		self.learn_rate = float(f.readline().strip())
		self.lr_multiplier = float(f.readline().strip())
		self.best_win_ratio = float(f.readline().strip())
		self.pure_mcts_playout_num = float(f.readline().strip())
		f.close()
			
	# agument data
	def get_equi_data(self, play_data):
		extend_data = []
		for state, mcts_prob, action, winner in play_data:
			for i in [1,2,3,4]:
				pi_board = mcts_prob.reshape(self.board_height, self.board_width)
				action_board = action.reshape(self.board_height, self.board_width)
				# rotate
				equi_state = np.rot90(state, i)
				equi_mcts_prob = np.rot90(pi_board, i)
				equi_action = np.rot90(action_board, i)
				extend_data.append((equi_state, equi_mcts_prob.flatten(), equi_action, winner))
				# flip
				equi_state = np.fliplr(equi_state)
				equi_mcts_prob = np.fliplr(equi_mcts_prob)
				equi_action = np.rot90(action_board, i)
				extend_data.append((equi_state, equi_mcts_prob.flatten(), equi_action, winner))
				
		return extend_data
		
	def collect_selfplay_data(self):
		winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
		play_data = list(play_data)[:]
		self.episode_len = len(play_data)
		#play_data = self.get_equi_data(play_data)
		#self.data_buffer.extend(play_data)
		return play_data
	
	def policy_update(self, mini_batch):
		#mini_batch = random.sample(self.data_buffer, self.batch_size)
		state_batch = np.float32([data[0] for data in mini_batch])
		mcts_probs_batch = np.float32([data[1] for data in mini_batch])
		action_batch = np.float32([data[2] for data in mini_batch])
		winner_batch = np.float32([data[3] for data in mini_batch])
		winner_batch = np.reshape(winner_batch, (-1, 1))
		old_probs, old_v = self.policy_value_net.policy_value(state_batch)
		for i in range(self.epochs):
			loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, action_batch, winner_batch, self.learn_rate * self.lr_multiplier)
			new_probs, new_v = self.policy_value_net.policy_value(state_batch)
			
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
		current_mcts_player = MCTSPlayer(
			self.policy_value_net.policy_value_fn,
			c_puct = self.c_puct,
			n_playout = self.n_playout,
			is_selfplay=False)
		pure_mcts_player =MCTS_Pure(
			c_puct = self.c_puct,
			n_playout = self.n_playout)
		win_count = defaultdict(int)
		for i in range(n_games):
			winner = self.game.start_play(
				current_mcts_player,
				pure_mcts_player,
				start_player=i%2,
				is_shown=0)
			win_count[winner]+=1
		win_ratio = 1.0*(win_count[P_X] + 0.5*win_count[P_E])/n_games
		print('num_playouts: {}, win: {}, loss: {}, tie: {}'.format(self.pure_mcts_playout_num, win_count[P_X], win_count[P_O], win_count[P_E]))
		return win_ratio
		
	def run(self):
		try:
			for i in range(self.game_batch_num):
				play_data = self.collect_selfplay_data()
				print('batch i:{}, episode_len:{}'.format(i+1, self.episode_len))
				print(self.board._state)
				loss, entropy = self.policy_update(play_data)
				if (i+1)%self.check_freg==0:
					print('current self-play batch: {}'.format(i+1))
					win_ratio = self.policy_evaluate()
					self.policy_value_net.save_model(CURRENT_MODEL_PATH)
					if win_ratio>self.best_win_ratio or win_ratio>0.99:
						self.best_win_ratio = win_ratio
						print('New best policy!!!')
						self.policy_value_net.save_model(BEST_MODEL_PATH)
						self.write_cache(CACHE_FILE_PATH)
						if self.best_win_ratio>0.99 and self.pure_mcts_playout_num<10000:
							self.pure_mcts_playout_num+=1000
							self.best_win_ratio = 0.5
		except KeyboardInterrupt:
			self.write_cache(CACHE_FILE_PATH)
			print('\n\rQuit')
			
if __name__ == '__main__':
	training_pipleline = TrainPipeline(resume=True)
	training_pipleline.run()