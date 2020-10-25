# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import numpy as np
from constants import P_X, P_O, P_E

class Board:
	def __init__(self, width=8, height=8, n_in_row=5):
		self.width = width
		self.height = height
		self.n_in_row = n_in_row
		
		self._state = np.ones([self.height, self.width], dtype=np.int8) * P_E
		self.availables = list(range(self.height*self.width))
		self.players = [P_X, P_O]
		
		# save the result and winner
		self._end = False
		self._winner = P_E # tie
		
	def init_board(self, start_player=0):
		self._state = np.ones([self.height, self.width], dtype=np.int8) * P_E
		
		self.current_player = self.players[start_player]
		self.availables = list(range(self.width * self.height))
		
		self._end = False
		self._winner = P_E # tie
		
	def _get_result(self, move):
		y, x = self.move_to_location(move)
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
			self._end = True
			self._winner = P_E
			return True, P_E
		
		return False, P_E
	
	def current_state(self):
		return self._state * self.current_player
		
	def move_to_location(self, move):
		h = move//self.width
		w = move%self.width
		return [h, w]
		
	def location_to_move(self, location):
		h = location[0]
		w = location[1]
		move = h*self.width + w
		return move
	
	def do_move(self, move):
		h, w = self.move_to_location(move)
		self._state[h, w] = self.current_player
		self.availables.remove(move)
		self._end, self._winner = self._get_result(move)
		self.current_player = P_X + P_O - self.current_player
		
	def undo_move(self, move):
		h, w = self.move_to_location(move)
		self._state[h, w] = P_E
		self.availables.append(move)
		self._end, self._winner = False, P_E
		self.current_player = P_X + P_O - self.current_player
		
	def game_end(self):
		return self._end, self._winner
		
class Game:
	def __init__(self, board):
		self.board =  board
		
	def graphic(self, board, player1, player2):
		width = board.width
		height = board.height

		print("Player", player1, "with X".rjust(3))
		print("Player", player2, "with O".rjust(3))
		print()
		s = ''
		chars = {P_X: 'x', P_E: '-', P_O: 'o'}
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
		with_human = False
		if player1.is_human or player2.is_human:
			with_human = True
			
		if is_shown:
			self.graphic(self.board, player1.player,  player2.player)
		while True:
			current_player = self.board.current_player
			player_in_turn = players[current_player]
			if with_human:
				move = player_in_turn.get_action(self.board)
			else:
				move = player_in_turn.get_action(self.board)
			self.board.do_move(move)
			if is_shown:
				self.graphic(self.board, player1.player, player2.player)
			end, winner = self.board.game_end()
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
			end, winner = self.board.game_end()
			if end:
				winners_z = np.zeros(len(current_players))
				if winner != P_E:
					winners_z[np.array(current_players)==winner] = 1.0
					winners_z[np.array(current_players)!=winner] = -1.0
				print('Current players', current_players)
				player.reset_player()
				if is_shown:
					if winner != P_E:
						print('Game end. Winner is player:', winner)
					else:
						print('game end.Tie')
				return winner, zip(states, mcts_probs, winners_z)