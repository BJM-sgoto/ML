# inference : https://github.com/junxiaosong/AlphaZero_Gomoku
import numpy as np
from operator import itemgetter
import math # faster than numpy for numbers
from constants import P_E, VIRTUAL_LOSS, NUM_INSTANCE, C_PUCT

def roll_out_fn(board): 
	valid_actions = board.availables
	action_probs = np.random.rand(len(valid_actions))
	return valid_actions, action_probs
	
def policy_value_fn(board):
	valid_actions = board.availables
	action_probs = np.ones(len(valid_actions))/ len(valid_actions)
	return zip(valid_actions, action_probs), 0
	
class TreeNode:
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
		if len(self._children)==0:
			self._child_ids = valid_actions.copy()
			self._child_P = action_priors
			self._child_Q = np.zeros_like(action_priors)
			self._child_n_visits = np.zeros_like(action_priors)
			for i in range(len(valid_actions)):
				self._children.append(TreeNode(self, valid_actions[i], i))
				
	# implement parallel mcts
	def select_multiple(self, num=1, virtual_loss=0.05):
		actions_s = []
		nodes = []
		for i in range(num):
			actions = []
			node = self
			while True:
				if len(node._children)==0:
					break
				max_child_pos = np.argmax((node._child_Q + C_PUCT * node._child_P * math.sqrt(node._n_visits))/(1 + node._child_n_visits))
				node._n_visits += 1
				node._child_n_visits[max_child_pos] += 1
				node._child_Q[max_child_pos] -= virtual_loss
				action = node._child_ids[max_child_pos]
				actions.append(action)
				
				node = node._children[max_child_pos]
				'''
				action, node = max(node._children.items(), key=lambda act_node: act_node[1].get_value())
				actions.append(action)
				node._n_visits += 1
				node._sqrt_n_visits = math.sqrt(node._n_visits)
				node._Q -= virtual_loss
				'''
			actions_s.append(actions)
			nodes.append(node)
		return nodes, actions_s	
		
	def update(self, pos_in_parent, child_value, virtual_loss=0.05):
		# child_id = self._child_ids.index(child_id)
		self._child_Q[pos_in_parent] += child_value + virtual_loss
		
	def backpropagate(self, child_value, virtual_loss=0.05):
		# only call this function from leaf node
		pos_in_parent = self._pos_in_parent
		node = self._parent
		
		if node:
			node.update(pos_in_parent, child_value, virtual_loss)
			child_value = -child_value
			pos_in_parent = node._pos_in_parent
			node = node._parent			
		
class MCTS:	
	def __init__(self, policy_value_fn, n_playout=2000):
		self._root = TreeNode(None, -1, -1)
		self._policy = policy_value_fn
		self._n_playout = n_playout // NUM_INSTANCE
	
	def _playout(self, board):
		# store actions
		actions = []
		
		# find the leaf node
		node = self._root
		nodes, actions_s = node.select_multiple(num=NUM_INSTANCE, virtual_loss=VIRTUAL_LOSS)
		ori_state = board.export_state()
		ends = [None for i in range(NUM_INSTANCE)]
		for i in range(NUM_INSTANCE):
			actions = actions_s[i]
			node = nodes[i]
			for action in actions:
				board.do_move(action)
			end, winner = board.get_result()
			if not end:
				action_probs, _ = self._policy(board)
				node.expand(action_probs)
			leaf_value = self._evaluate_rollout(board)
			node.backpropagate(-leaf_value, virtual_loss=VIRTUAL_LOSS)
			board.import_state(ori_state)
	
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
		return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]
	
	def update_with_move(self, last_move):
		if last_move in self._root._children:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, -1, -1)
			
class MCTSPlayer:
	def __init__(self, n_playout=2000):
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