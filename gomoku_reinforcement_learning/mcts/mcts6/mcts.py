# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import numpy as np
from mcts_pure import MCTS as BaseMCTS, MCTSPlayer as BaseMCTSPlayer
from constants import P_E

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs = probs / np.sum(probs)
	return probs
	
class MCTS(BaseMCTS):
	def __init__(self, policy_value_fn, c_puct=5.0, n_playout=10000):
		super(MCTS, self).__init__(policy_value_fn, c_puct, n_playout)
		self._policy = policy_value_fn
		self._c_puct = c_puct
		self._n_playout = n_playout
	
	def _playout(self, board):
		# store actions
		actions = []
		
		# playout
		node = self._root
		while True:
			if node.is_leaf():
				break
			action, node = node.select(self._c_puct)
			actions.append(action)
			board.do_move(action)
		action_probs, leaf_value = self._policy(board) # do not play to the end of the game
		
		end, winner = board.game_end()
		if not end:
			node.expand(action_probs)
		else:
			if winner == P_E: # tie
				leaf_value = 0.0
			else:
				leaf_value = 1.0 if winner == board.current_player else -1.0
		node.update_recursive(-leaf_value) #???
		
		# restore state
		for action in reversed(actions):
			board.undo_move(action)

	# use deterministic action
	def get_move_probs(self, board, temp=1e-3):
		for n in range(self._n_playout):
			self._playout(board)
		act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
		acts, visits = zip(*act_visits)
		act_probs = softmax(np.log(np.array(visits) + 1e-10)/temp)
		return acts, act_probs
		
class MCTSPlayer(BaseMCTSPlayer):
	def __init__(self, policy_value_fn, c_puct=5.0, n_playout=2000, is_selfplay=False):
		self.mcts  = MCTS(policy_value_fn, c_puct, n_playout)
		self._is_selfplay = is_selfplay
		self.is_human = False
		
	def get_action(self, board, temp=1e-3, return_prob=False):
		sensible_moves = board.availables
		move_probs = np.zeros(board.width * board.height, dtype=np.float32)
		if len(sensible_moves)>0:	
			acts, probs = self.mcts.get_move_probs(board, temp)
			move_probs[list(acts)] = probs
			if self._is_selfplay:
				move = np.random.choice(
					acts,
					p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
				self.mcts.update_with_move(move)
			else:
				move = np.random.choice(acts, p=probs)
				self.mcts.update_with_move(-1)
			
			if return_prob:
				return move, move_probs
			else:
				return move
		else:
			print('Warning: board is full')
			
	def get_instant_action(self, board):
		action_probs, value = self.mcts._policy(board)
		action = max(action_probs, key = lambda item: item[1])[0]
		return action