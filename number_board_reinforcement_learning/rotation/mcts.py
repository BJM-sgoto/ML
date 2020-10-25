# inference : https://github.com/junxiaosong/AlphaZero_Gomoku
import numpy as np
from operator import itemgetter
from env_rotation import REWARD_NOT_SOLVED, REWARD_SOLVED

class TreeNode:
	# prior : prior probability to pick this action
	def __init__(self, parent, prior_p):
		self._parent = parent
		self._children = {}
		self._n_visits = 0
		
		self._P = prior_p
		self._U = 0
		self._W = 0.0
		
	def expand(self, action_priors):
		for action, prob in action_priors:
			if action not in self._children:
				self._children[action] = TreeNode(self, prob)
				
	def select(self, c_puct=1.414):
		return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)) # pick the child with highest u
		
	def update(self, value):
		self._n_visits += 1
		self._W = max(value, self._W)
		
	def update_recursive(self, value):
		self.update(value)
		if self._parent:
			self._parent.update_recursive(self._W)
		
	def get_value(self, c_puct):
		self._U = c_puct * self._P * np.sqrt(self._parent._n_visits)/(1 + self._n_visits)
		return self._U + self._W
	
	def is_leaf(self):
		return len(self._children)==0
		
	def is_root(self):
		return self._parent is None
		
class MCTS:	
	def __init__(self, policy_value_fn, c_puct=5.0, n_playout=2000):
		self._root = TreeNode(None, 1.0)
		self.policy_value_fn = policy_value_fn
		self._c_puct = c_puct
		self._n_playout = n_playout
	
	# return list of actions and answer found the path or not
	def _playout(self, env):
		# store actions
		actions = []
		found_path = False
		# find the leaf node
		node = self._root
		while True:
			if node.is_leaf():
				break
			action, node = node.select(self._c_puct)
			actions.append(action)
			env.step(action)
		action_probs, leaf_value = self.policy_value_fn(env)
		action_probs = zip(range(action_probs.shape[0]), action_probs)
		reward = env.get_result()
		
		if reward == REWARD_NOT_SOLVED:
			node.expand(action_probs)
		else:
			# TODO: output path
			leaf_value = REWARD_SOLVED
			found_path = True

		node.update_recursive(leaf_value)
		
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
		actions = [max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]]
		return actions, found_path
	
	def update_with_move(self, last_move):
		if last_move in self._root._children:
			self._root = self._root._children[last_move]
			self._root._parent = None
		else:
			self._root = TreeNode(None, 1.0)