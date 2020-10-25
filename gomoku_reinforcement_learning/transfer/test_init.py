from mcts import MCTSPlayer
from neural_net import PolicyValueNet
from test import Human
from constants import BOARD_WIDTH, BOARD_HEIGHT, N_IN_ROW, TRAIN_MODEL_NAME, TRAIN_MODEL_PATH
import tensorflow.compat.v1 as tf
from board import Board, Game

tf.disable_v2_behavior()
tf.reset_default_graph()

def run():
	n = N_IN_ROW
	width, height = BOARD_WIDTH, BOARD_HEIGHT
	session = tf.Session()
	try:
		board = Board(width=width, height=height, n_in_row=n)
		game = Game(board)
		best_policy = PolicyValueNet(
			session, 
			TRAIN_MODEL_NAME, 
			width, 
			height, 
			model_file=TRAIN_MODEL_PATH)
		mcts_player = MCTSPlayer(best_policy.policy_value_fn,
								 n_playout=5000) 
		human = Human()
		game.start_play(human, mcts_player, start_player=1, is_shown=1)
	except KeyboardInterrupt:
		print('\n\rquit')


if __name__ == '__main__':
	run()