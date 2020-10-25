# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import numpy as np
from constants import P_X, P_O, P_E

class Board:
	def __init__(self, width=8, height=8, n_in_row=5):
		self.width = width
		self.height = height
		self.n_in_row = n_in_row
		
		self._state = np.ones([self.height, self.width], dtype=np.int8) * P_E
		self._moves = []
		self._initial_availables = set(range(self.height*self.width))
		self.availables = list(self._initial_availables)
		self.players = [P_X, P_O]		
		
		
	def init_board(self, start_player=0):
		self._state = np.ones([self.height, self.width], dtype=np.int8) * P_E
		self._moves.clear()
		self.current_player = self.players[start_player]
		self.availables = list(self._initial_availables)
	
	def get_result(self):
		if len(self._moves)<2*self.n_in_row:
			return False, P_E
			
		move = self._moves[-1]
		y = move//self.width
		x = move%self.width
		player = self._state[y, x]
		
		# check horizontal cells
		score = 1
		min_x = max(-1, x - self.n_in_row)
		for i in range(x-1, min_x, -1):
			if self._state[y, i] == player:
				score += 1
			else:
				break
		max_x = min(self.width, x + self.n_in_row)
		for i in range(x + 1, max_x):
			if self._state[y, i] == player:
				score += 1
			else:
				break
			
		if score >=	self.n_in_row:
			return True, player
		
		# check vertical cells
		score = 1
		min_y = max(-1, y - self.n_in_row)
		for i in range(y - 1, min_y, -1):
			if self._state[i, x] == player:
				score += 1
			else:
				break
		max_y = min(self.height, y + self.n_in_row)
		for i in range(y + 1, max_y):
			if self._state[i, x] == player:
				score += 1
			else:
				break
		
		if score>=self.n_in_row:
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
				
		if score >= self.n_in_row:
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
		
		if score >= self.n_in_row:
			return True, player
		
		if len(self.availables)<=0:
			return True, P_E
		
		return False, P_E
	
	def current_state(self):
		return self._state * self.current_player
		
	def location_to_move(self, location):
		h = location[0]
		w = location[1]
		move = h*self.width + w
		return move
	
	def do_move(self, move):
		h = move//self.width
		w = move%self.width
		self._state[h, w] = self.current_player
		self._moves.append(move)
		self.availables.remove(move)
		self.current_player = P_X + P_O - self.current_player
	
	# move without update internal states. ex: availables
	# => after calling this function, we must validate internal state by calling self.validate()
	def do_simple_move(self, move):
		y = move//self.width
		x = move%self.width
		self._state[y, x] = self.current_player
		self._moves.append(move)
		self.current_player = P_X + P_O - self.current_player
		
	def validate(self):
		self.availables = list(self._initial_availables - set(self._moves))
		
	def undo_move(self, move):
		h = move//self.width
		w = move%self.width
		self._state[h, w] = P_E
		self._moves.pop()
		self.availables.append(move)
		self.current_player = P_X + P_O - self.current_player
	
	def export_state(self):
		return (
			self._state.copy(),
			self._moves.copy(),
			self.availables.copy(),
			self.current_player)
		
	def import_state(self, state):
		self._state = state[0].copy()
		self._moves = state[1].copy()
		self.availables = state[2].copy()
		self.current_player = state[3]
		
class Game:
	def __init__(self, board):
		self.board =  board
		
	def graphic(self, board, player1, player2):
		width = board.width
		height = board.height

		print('--------------------------')
		s = ''
		chars = {P_X: '1', P_E: '0', P_O: '2'}
		for i in range(height):
			for j in range(width):
				s+=chars[board._state[i,j]] + '  '
			s +='\r\n'
		print(s)

	def start_play(self, player1, player2, start_player=0, is_shown=1):
		self.board.init_board(start_player)
		p1, p2 = self.board.players
		player1.set_player_ind(p1)
		player2.set_player_ind(p2)
		players = {p1: player1, p2: player2}
			
		if is_shown:
			self.graphic(self.board, player1.player,  player2.player)
		while True:
			current_player = self.board.current_player
			player_in_turn = players[current_player]
			move = player_in_turn.get_action(self.board)
			self.board.do_move(move)
			if is_shown:
				self.graphic(self.board, player1.player, player2.player)
			end, winner = self.board.get_result()
			if end:
				if is_shown:
					if winner != P_E:
						print('Game end. Winner is', winner)
					else:
						print('Game end. Tie')
				return winner
	
	def start_self_play(self, player, is_shown=0, temp=1e-3):
		# deterministic action
		# do not show
		self.board.init_board()
		p1, p2 = self.board.players
		states, mcts_probs, current_players = [], [], []
		while True:
			move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
			states.append(self.board.current_state())
			mcts_probs.append(move_probs)
			current_players.append(self.board.current_player)
			self.board.do_move(move)
			if is_shown:
				self.graphic(self.board, p1, p2)
			end, winner = self.board.get_result()
			if end:
				winners_z = np.zeros(len(current_players))
				if winner != P_E:
					winners_z[np.array(current_players)==winner] = 1.0
					winners_z[np.array(current_players)!=winner] = -1.0
				player.reset_player()
				if is_shown:
					if winner != P_E:
						print('Game end. Winner is player:', winner)
					else:
						print('game end.Tie')
				return winner, zip(states, mcts_probs, winners_z)