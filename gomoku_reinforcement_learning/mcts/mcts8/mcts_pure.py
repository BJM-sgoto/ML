# inference : https://github.com/junxiaosong/AlphaZero_Gomoku
# TODO: what is dirichlet noise/distribution
import numpy as np
from operator import itemgetter
from constants import P_E, NUM_ACTION, C_PUCT, VIRTUAL_LOSS
import math

def roll_out_fn(board): 
	valid_actions = board.availables
	action_probs = np.random.rand(len(valid_actions))
	return zip(valid_actions, action_probs)
	
def policy_value_fn(board):
	valid_actions = board.availables
	action_probs = np.ones(len(valid_actions))/ len(valid_actions)
	return action_probs, 0
	
class TreeNode:
	# init p, q???
	def __init__(self, parent, move, pos_in_parent):
		# parent
		self._parent = parent
		
		# current node properties
		self._n_visits = 0
		self._move = move
		self._pos_in_parent = pos_in_parent		
		
		# children properties
		self._children = []
		self._child_ids = None
		self._child_n_visits = None
		self._child_Q = None
		self._child_P = None
		
	def expand(self, valid_actions, action_priors):
		# only expand nodes that are truly leaf nodes
		if len(self._children)==0:
			self._child_ids = valid_actions.copy()
			self._child_P = action_priors
			self._child_Q = np.zeros_like(action_priors)
			self._child_n_visits = np.zeros_like(action_priors)
			for i in range(len(valid_actions)):
				self._children.append(TreeNode(self, valid_actions[i], i))
				
	def select(self, virtual_loss=0.05):
		actions = []
		node = self
		while True:
			if len(node._children)==0:
				break
			max_child_pos = np.argmax((node._child_Q + C_PUCT * node._child_P * math.sqrt(node._n_visits))/(1+node._child_n_visits))
			# change node properties
			node._n_visits += 1
			# change children properties
			node._child_n_visits[max_child_pos] += 1
			node._child_Q[max_child_pos] -= virtual_loss
			action = node._child_ids[max_child_pos]
			actions.append(action)
			# move downto child
			node = node._children[max_child_pos]
		return node, actions
	
	def backpropagate(self, child_value, virtual_loss=0.05):
		pos_in_parent = self._pos_in_parent
		node = self._parent
		if node:
			node._child_Q[pos_in_parent] += child_value + virtual_loss
			child_value = -child_value
			pos_in_parent = node._pos_in_parent
			node = node._parent
	
class MCTS:	
	def __init__(self, policy_value_fn, n_playout=2000):
		self._root = TreeNode(None, -1, -1)
		self._policy_value_fn = policy_value_fn
		self._n_playout = n_playout
	
	def _playout(self, board):
		# find the leaf node
		node, actions = self._root.select(virtual_loss=VIRTUAL_LOSS)
		for action in actions:
			board.do_move(action)
			
		end, winner = board.get_result()
		if not end:
			action_probs, _ = self._policy_value_fn(board)
			valid_actions = board.availables
			node.expand(valid_actions, action_probs)
		leaf_value = self._evaluate_rollout(board)
		node.backpropagate(-leaf_value, virtual_loss=VIRTUAL_LOSS)
		
		# restore state
		for action in reversed(actions):
			board.undo_move(action)
	
	def _evaluate_rollout(self, board, limit=1000):
		# store actions
		actions = []
		
		# play until game ends
		player = board.current_player
		for i in range(limit):	
			end, winner = board.get_result()
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
		best_child_pos = np.argmax(self._root._child_n_visits)
		return self._root._child_ids[best_child_pos]
	
	def update_with_move(self, last_move):
		if last_move in self._root._children:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, -1, -1)
			
class MCTSPlayer:
	def __init__(self, c_puct=5.0, n_playout=2000):
		self.mcts = MCTS(policy_value_fn, n_playout)
		
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