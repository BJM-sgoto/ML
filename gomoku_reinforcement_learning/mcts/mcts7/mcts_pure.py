# inference : https://github.com/junxiaosong/AlphaZero_Gomoku
# TODO: what is dirichlet noise/distribution
import numpy as np
from operator import itemgetter
from constants import P_E

def roll_out_fn(board): 
	valid_actions = board.availables
	action_probs = np.random.rand(len(valid_actions))
	return zip(valid_actions, action_probs)
	
def policy_value_fn(board):
	valid_actions = board.availables
	action_probs = np.ones(len(valid_actions))/ len(valid_actions)
	return zip(valid_actions, action_probs), 0
	
class TreeNode:
	# init p, q???
	def __init__(self, parent, prior_p):
		self._parent = parent
		self._children = {}
		self._n_visits = 0
		self._Q = 0
		self._P = prior_p
		self._U = 0
		
	def expand(self, action_priors):
		for action, prob in action_priors:
			if action not in self._children:
				self._children[action] = TreeNode(self, prob)
				
	def select(self, c_puct):
		return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)) # pick the child with highest u
		
	def update(self, leaf_value):
		self._n_visits += 1
		self._Q += (leaf_value - self._Q) / self._n_visits
		
	def update_recursive(self, leaf_value):
		if self._parent:
			self._parent.update_recursive(-leaf_value)
		self.update(leaf_value)
		
	def get_value(self, c_puct):
		self._U = c_puct * self._P * np.sqrt(self._parent._n_visits)/(1 + self._n_visits)
		return self._Q + self._U
	
	def is_leaf(self):
		return len(self._children)==0
		
	def is_root(self):
		return self._parent is None
		
class MCTS:	
	def __init__(self, policy_value_fn, c_puct=5.0, n_playout=2000):
		self._root = TreeNode(None, 1.0)
		self._policy = policy_value_fn
		self._c_puct = c_puct
		self._n_playout = n_playout
	
	def _playout(self, board):
		# store actions
		actions = []
		
		# find the leaf node
		node = self._root
		while True:
			if node.is_leaf():
				break
			action, node = node.select(self._c_puct)
			actions.append(action)
			board.do_move(action)
		action_probs, _ = self._policy(board)
		end, winner = board.game_end()
		if not end:
			node.expand(action_probs)
			
		leaf_value = self._evaluate_rollout(board)
		node.update_recursive(-leaf_value)
		
		# restore state
		for action in reversed(actions):
			board.undo_move(action)
	
	def _evaluate_rollout(self, board, limit=1000):
		# store actions
		actions = []
		
		# play until game ends
		player = board.current_player
		for i in range(limit):	
			end, winner = board.game_end()
			if end:
				break
			action_probs = roll_out_fn(board)
			max_action = max(action_probs, key=itemgetter(1))[0]
			actions.append(max_action)
			board.do_move(max_action)
		
		# restore state
		for action in reversed(actions):
			board.undo_move(action)
		
		if winner==P_E:
			return 0
		else:
			return 1 if winner==player else -1 #???
			
	def get_move(self, board):
		'''
		run all playouts and return most visited action
		'''
		for n in range(self._n_playout):
			self._playout(board)
		return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]
	
	def update_with_move(self, last_move):
		if last_move in self._root._children:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, 1.0)
			
class MCTSPlayer:
	def __init__(self, c_puct=5.0, n_playout=2000):
		self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
		self.is_human = False
		
	def set_player_ind(self, p):
		self.player = p
	
	def reset_player(self):
		self.mcts.update_with_move(-1)
	
	def get_action(self, board):
		sensible_moves = board.availables
		if len(sensible_moves)>0:
			move = self.mcts.get_move(board)
			self.mcts.update_with_move(-1)
			return move
		else:
			print('Warning: board is full')