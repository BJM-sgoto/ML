# reference https://www.moderndescartes.com/essays/deep_dive_mcts/
# implement: 	- improve numpy implementation
# 						- parallel mcts
import numpy as np
from env_cube import REWARD_NOT_SOLVED, REWARD_SOLVED

N_ACTION = 12
VIRTUAL_LOSS = 0.1
C_PUCT = 3.0
class TreeNode:
	# prior : prior probability to pick this action
	count = 0
	def __init__(self, move, parent):
		self._move = move
		self._parent = parent
		self._children = {}
		self._n_visits = 0
		self._value = 0
		
		self._child_n_visits = np.zeros([N_ACTION], dtype=np.float32)
		self._child_priors = np.ones([N_ACTION], dtype=np.float32) / N_ACTION
		self._child_value = np.zeros([N_ACTION], dtype=np.float32)
	
	@property
	def number_visits(self):
		return self._n_visits
		
	@number_visits.setter
	def number_visits(self, value):
		self._n_visits = value
		if self._parent is not None:
			self._parent._child_n_visits[self._move] = value
		
	@property
	def total_value(self):
		return self._value
	
	@total_value.setter
	def total_value(self, value):
		self._value = value
		if self._parent is not None:
			self._parent._child_value[self._move] = value
	
	def child_Q(self):
		return self._child_value
		
	def child_U(self):
		return C_PUCT * np.sqrt(self.number_visits) * self._child_priors/(1 + self._child_n_visits)
	
	def child_UCB(self):
		return self.child_Q() + self.child_U()
	
	def best_child(self): # select
		return np.argmax(self.child_UCB())
		
	def select_leaf(self):
		node = self
		actions = []
		while not node.is_leaf():
			node.number_visits+=1
			node.total_value-=VIRTUAL_LOSS
			action = node.best_child()
			actions.append(action)
			node = node._children[action]
		return actions, node
	
	def backup(self, value_estimate):
		node = self
		# TODO : node is not None
		while node._parent is not None:
			# node.number_visits+=1 # no need to add 1 because we already added 1 when select leaf
			node.total_value+=VIRTUAL_LOSS# recover total value
			node.total_value = max(node.total_value, value_estimate)
			node = node._parent
			
	def expand(self, child_priors):
		self._child_priors = child_priors
		for move in range(len(child_priors)):
			if move not in self._children:
				self._children[move] = TreeNode(move, self)
			
	def is_leaf(self):
		return len(self._children)==0
		
	def is_root(self):
		return self._parent is None

class Quick_MCTS:
	def __init__(self, policy_value_fn, n_playout=2000):
		self._root = TreeNode(-1, None)
		self._policy_value_fn = policy_value_fn
		self._n_playout = n_playout
		
	def _playout(self, env):
		# store actions
		actions = []
		found_path = False
		# TODO: use parallel implemention here
		# find the leaf node
		actions, leaf_node = self._root.select_leaf()
		for action in actions:
			env.step(action)
		reward = env.get_result()
		found_path = False
		
		leaf_value = reward
		if reward==REWARD_SOLVED:
			found_path = True			
		else:
			action_probs, leaf_value = self._policy_value_fn(env)
			leaf_node.expand(action_probs)
		
		leaf_node.backup(leaf_value)
		# restore state
		for action in reversed(actions):
			env.undo(action)
		return actions, found_path
	
	def get_move(self, env):
		found_path = False
		for n in range(self._n_playout):
			actions, found_path = self._playout(env)
			if found_path:
				break
		if found_path:
			return actions, found_path
		actions = [np.argmax(self._root._child_n_visits)]
		return actions, found_path
		
	def update_with_move(self, move):
		if move in self._root._children:
			self._root = self._root._children[move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, 1.0)