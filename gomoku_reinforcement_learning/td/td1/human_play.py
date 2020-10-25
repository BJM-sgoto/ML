# -*- coding: utf-8 -*-

from __future__ import print_function
from board import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts import MCTSPlayer
from neural_net import PolicyValueNet # Tensorflow
from constants import BOARD_WIDTH, BOARD_HEIGHT, N_IN_ROW

class Human:
	def __init__(self):
		self.player = None
		self.is_human = True

	def set_player_ind(self, p):
		self.player = p
	
	def get_action(self, board):
		try:
			location = input("Your move: ")
			if isinstance(location, str):  # for python3
				location = [int(n, 10) for n in location.split(",")]
			move = board.location_to_move(location)
		except Exception as e:
			move = -1
		if move == -1 or move not in board.availables:
			print("invalid move")
			move = self.get_action(board)
		return move
		
	def get_instant_action(self, board):
		return self.get_action(board)
	

def run():
	n = N_IN_ROW
	width, height = BOARD_WIDTH, BOARD_HEIGHT
	model_file = './model/best_model'
	try:
		board = Board(width=width, height=height, n_in_row=n)
		game = Game(board)
		best_policy = PolicyValueNet(width, height, model_file = model_file)
		mcts_player = MCTSPlayer(best_policy.policy_value_fn,
								 c_puct=5,
								 n_playout=5000) 
		human = Human()
		game.start_play(human, mcts_player, start_player=1, is_shown=1)
	except KeyboardInterrupt:
		print('\n\rquit')


if __name__ == '__main__':
	run()