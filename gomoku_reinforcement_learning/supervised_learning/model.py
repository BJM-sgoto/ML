import tensorflow.compat.v1 as tf
import numpy as np
import random

tf.disable_v2_behavior()
tf.reset_default_graph()
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

BOARD_SIZE = 15
N_IN_ROW = 5

DATA_FILE = './data.txt'
P_X = 1
P_O = -1
P_E = 0

NUM_EPOCH = 100
BATCH_SIZE = 20 # 

class SimpleBoard:
	def __init__(self):
		self.reset()
		
	def reset(self):
		self.board = np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.int8) * P_E
		self.current_player = np.random.choice([P_X, P_O])
	
	def reset_with_random_move(self):
		self.reset()
		random_move = [np.random.randint(low=0, high=BOARD_SIZE), np.random.randint(low=0, high=BOARD_SIZE)]
		self.do_move(random_move)
	
	def do_move(self, move):
		self.last_move = move
		self.board[move[0], move[1]] = self.current_player
		self.current_player = P_O + P_X - self.current_player
	
	def current_state(self):
		return self.board * self.current_player
	
	def encode_move(self, move):
		encoded_move = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
		encoded_move[move[0], move[1]] = 1
		return encoded_move
	
	def gen_episode(self, episode):
		episode = np.transpose(episode, [1,0])
		self.reset()
		boards = []
		moves = []
		for move in episode:
			_state = self.current_state()
			boards.append(self.current_state())
			_move = self.encode_move(move)
			moves.append(self.encode_move(move))
			self.do_move(move)
		return boards, moves
	
	def get_result(self):
		y, x = self.last_move
		player = self.board[y, x]
		
		# check horizontal cells
		score = 1
		min_x = max(-1, x - N_IN_ROW)
		for i in range(x-1, min_x, -1):
			if self.board[y, i] == player:
				score += 1
			else:
				break
		max_x = min(BOARD_SIZE, x + N_IN_ROW)
		for i in range(x + 1, max_x):
			if self.board[y, i] == player:
				score += 1
			else:
				break
			
		if score >=	N_IN_ROW:
			return True, player
		
		# check vertical cells
		score = 1
		min_y = max(-1, y - N_IN_ROW)
		for i in range(y - 1, min_y, -1):
			if self.board[i, x] == player:
				score += 1
			else:
				break
		max_y = min(BOARD_SIZE, y + N_IN_ROW)
		for i in range(y + 1, max_y):
			if self.board[i, x] == player:
				score += 1
			else:
				break
		
		if score>=N_IN_ROW:
			return True, player
			
		# check NS->ES diagonal cells
		score = 1
		d = min(x - min_x, y - min_y)
		for i in range(-1, -d, -1):
			if self.board[y+i, x+i]==player:
				score += 1
			else:
				break
			
		d = min(max_x - x, max_y - y)
		for i in range(1, d):
			if self.board[y+i, x+i]==player:
				score += 1
			else:
				break
				
		if score >= N_IN_ROW:
			return True, player
			
		# check NE->WS diagonal cells
		score = 1
		d = min(max_x - x, y - min_y)
		for i in range(1, d):
			if self.board[y-i, x+i]==player:
				score += 1
			else:
				break
				
		d = min(x - min_x, max_y - y)
		for i in range(1, d):
			if self.board[y+i, x-i]==player:
				score += 1
			else:
				break
				
		if score >= N_IN_ROW:
			return True, player
		
		ys, xs = np.where(self.board==P_E)
		if len(ys)<=0:
			return True, P_E
		
		return False, P_E
		
	def to_string(self):
		s = ''
		for i in range(-1, BOARD_SIZE):
			if i == -1:
				for j in range(-1, BOARD_SIZE):
					if j==-1:
						s += '  '
					else:
						s += str(j%10) + ' '
			else:
				for j in range(-1, BOARD_SIZE):
					if j == -1:
						s += str(i%10) + ' '
					elif self.board[i,j]==P_X:
						s += 'x '
					elif self.board[i,j]==P_O:
						s += 'o '
					else:
						s += '- '
			s += '\r\n'
		return s

class Model:
	def __init__(self):
		self.reuse = False
		self.board = SimpleBoard()
		
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
		for move_y, move_x in dataset:
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
			
			extended_dataset.append(np.int32([move_y, move_x]))
			extended_dataset.append(np.int32([move_y_90, move_x_90]))
			extended_dataset.append(np.int32([move_y_180, move_x_180]))
			extended_dataset.append(np.int32([move_y_270, move_x_270]))
			
			# exchange move_x and move_y 
			extended_dataset.append(np.int32([move_x, move_y]))
			extended_dataset.append(np.int32([move_x_90, move_y_90]))
			extended_dataset.append(np.int32([move_x_180, move_y_180]))
			extended_dataset.append(np.int32([move_x_270, move_y_270]))
		return extended_dataset
		
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
		
	def make_model(self, boards, training=False):
		feature = tf.cast(tf.reshape(boards, [-1, BOARD_SIZE, BOARD_SIZE, 1]), dtype=tf.float32)
		with tf.variable_scope('model', reuse=self.reuse):
			for i in range(7):
				pre_feature = feature
				feature = tf.layers.conv2d(
					feature,
					filters=64,
					kernel_size=(3,3),
					padding='same',
					activation=tf.nn.leaky_relu)
				feature = tf.layers.conv2d(
					feature,
					filters=64,
					kernel_size=(3,3),
					padding='same',
					activation=tf.nn.leaky_relu)
				feature = pre_feature + feature
				feature = tf.layers.batch_normalization(feature, training=training)
			feature = tf.layers.conv2d(
					feature,
					filters=4,
					kernel_size=(1,1),
					padding='same',
					activation=tf.nn.leaky_relu)
			feature = tf.layers.flatten(feature)
			feature = tf.layers.dense(
				feature,
				units=BOARD_SIZE*BOARD_SIZE,
				activation=tf.nn.sigmoid)
			
			valid_actions = tf.cast(tf.equal(boards, P_E), dtype=tf.float32)
			valid_actions = tf.layers.flatten(valid_actions)
			'''
			feature = feature - tf.reduce_max(feature, axis=1, keepdims=True)
			feature = tf.exp(feature) * valid_actions
			moves = feature / tf.reduce_max(feature, axis=1, keepdims=True)
			'''
			moves = feature *valid_actions
			moves = tf.reshape(moves, [-1, BOARD_SIZE, BOARD_SIZE])
			
			self.reuse = True
		return moves
	
	def make_train_data(self, episodes):
		boards = []
		moves = []
		for episode in episodes:
			_boards, _moves = self.board.gen_episode(episode)
			boards.extend(_boards)
			moves.extend(_moves)
		return np.int8(boards), np.float32(moves)
	
	def train(self, datafile='./data.txt', model_path='./model/model', resume=False):
		X = tf.placeholder(tf.int8, [None, BOARD_SIZE, BOARD_SIZE])
		Y = tf.placeholder(tf.float32, [None, BOARD_SIZE, BOARD_SIZE])
		PY = self.make_model(X, training=True)
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y - PY), axis=[1,2]))
		train_op = tf.train.AdamOptimizer().minimize(loss)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_op])
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		
		dataset = self.make_dataset(datafile)
		dataset = self.extend_dataset(dataset)
		count_to_save = 0
		for i in range(NUM_EPOCH):
			self.shuffle_dataset(dataset)
			n_data = len(dataset)
			for j in range(0, n_data, BATCH_SIZE):
				end_j = min(n_data, j + BATCH_SIZE)
				boards, moves = self.make_train_data(dataset[j: end_j])
				loss_val, _ = session.run([loss, train_op], feed_dict={X: boards, Y: moves})
				print('Epoch {:04d}, Progress {:04d}, Loss {:06f}'.format(i, j, loss_val))
				count_to_save += 1
				if count_to_save>=500:
					count_to_save = 0
					saver.save(session, model_path)
		session.close()
		
	def test(self, model_path='./model/model'):
		X = tf.placeholder(tf.int8, [None, BOARD_SIZE, BOARD_SIZE])
		PY = self.make_model(X, training=False)
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		self.board.reset_with_random_move()
		
		end, winner = self.board.get_result()
		print(self.board.to_string())
		while not end:
			if self.board.current_player == P_X: # human
				while True:
					try:
						move = input('You move: ')
						move = move.strip().split(',')
						move = [int(item) for item in move]
						self.board.do_move(move)
						break
					except KeyboardInterrupt:
						print('Quit')
						exit()
					except:
						print('invalid move, move must be of form y, x')
			else: # computer
				current_state = self.board.current_state()
				current_state = np.reshape(current_state, [-1, BOARD_SIZE, BOARD_SIZE])
				policy = session.run(PY, feed_dict={X: current_state})[0]
				max_val = np.max(policy)
				y, x = np.where(policy==max_val)
				y = y[0]
				x = x[0]
				print('Computer move {}, {}'.format(y,x))
				self.board.do_move([y,x])
			print(self.board.to_string())
			end, winner = self.board.get_result()
			print('End/ Winner', end, winner)
		if winner == P_O:
			print('You lose')
		elif winner == P_X:
			print('You win')
		session.close()	
			
model = Model()
'''
dataset = model.train(
	datafile='./data.txt', 
	model_path='./model/model', 
	resume=True)
'''
model.test(model_path='./model/model')