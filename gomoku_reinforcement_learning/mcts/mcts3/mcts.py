# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import numpy as np
import copy
from mcts_pure import TreeNode as BaseTreeNode, MCTS as BaseMCTS, MCTSPlayer as BaseMCTSPlayer

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs = probs / np.sum(probs)
	return probs
	
class TreeNode(BaseTreeNode):
	def __init__(self, parent, prior_p):
		super(TreeNode, self).__init__(parent, prior_p)

class MCTS(BaseMCTS):
	def __init__(self, policy_value_fn, c_puct=5.0, n_playout=10000):
		super(MCTS, self).__init__(policy_value_fn, c_puct, n_playout)
		self._policy = policy_value_fn
		self._c_puct = c_puct
		self._n_playout = n_playout
	
	def _playout(self, state):
		node = self._root
		while True:
			if node.is_leaf():
				break
			action, node = node.select(self._c_puct)
			state.do_move(action)
		action_probs, leaf_value = self._policy(state) # do not play to the end of the game
		end, winner = state.game_end()
		if not end:
			node.expand(action_probs)
		else:
			if winner == -1: # tie
				leaf_value = 0.0
			else:
				if winner == state.current_player:
					leaf_value = 1.0 # current player loses => update_recursive with -leaf_value
				else:
					leaf_value = -1.0
		# current 
		node.update_recursive(-leaf_value)
		
	# use deterministic action
	def get_move_probs(self, state, temp=1e-3):
		for n in range(self._n_playout):
			state_copy = copy.deepcopy(state)
			self._playout(state_copy)
		act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
		acts, visits = zip(*act_visits)
		act_probs = softmax(np.log(np.array(visits) + 1e-10)/temp)
		return acts, act_probs
		
class MCTSPlayer(BaseMCTSPlayer):
	def __init__(self, policy_value_fn, c_puct=5.0, n_playout=2000, is_selfplay=False):
		self.mcts  = MCTS(policy_value_fn, c_puct, n_playout)
		self._is_selfplay = is_selfplay
		
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