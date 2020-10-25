# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import numpy as np
from constants import P_E, NUM_ACTION, VIRTUAL_LOSS
from mcts_pure import MCTS as BaseMCTS, MCTSPlayer as BaseMCTSPlayer

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs = probs / np.sum(probs)
	return probs

class MCTS(BaseMCTS):
	def _playout(self, board):
		# find the leaf node
		node, actions = self._root.select(virtual_loss=VIRTUAL_LOSS)
		for action in actions:
			board.do_move(action)
			
		end, winner = board.get_result()
		if not end:
			action_probs, leaf_value = self._policy_value_fn(board)
			valid_actions = board.availables
			action_probs = action_probs[valid_actions]
			node.expand(valid_actions, action_probs)
		else:
			if winner==P_E:
				leaf_node=0.0
			else:
				leaf_value=1.0 if winner==board.current_player else -1.0
		leaf_value = self._evaluate_rollout(board)
		node.backpropagate(-leaf_value, virtual_loss=VIRTUAL_LOSS)
		
		# restore state
		for action in reversed(actions):
			board.undo_move(action)
		
	# use deterministic action
	def get_move_probs(self, board, temp=1e-3):
		for n in range(self._n_playout):
			self._playout(board)
		acts = self._root._child_ids
		visits = self._root._child_n_visits
		act_probs = softmax(np.log(visits + 1e-10)/temp)
		return acts, act_probs
		
class MCTSPlayer(BaseMCTSPlayer):
	def __init__(self, policy_value_fn, n_playout=2000, is_selfplay=False):
		self.mcts = MCTS(policy_value_fn, n_playout)
		self._is_selfplay = is_selfplay

	def get_action(self, board, temp=1e-3, return_prob=False):
		sensible_moves = board.availables
		move_probs = np.zeros(board.width * board.height, dtype=np.float32)
		if len(sensible_moves)>0:	
			acts, probs = self.mcts.get_move_probs(board, temp)
			move_probs[list(acts)] = probs
			if self._is_selfplay:
				probs = 0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
				probs = probs / np.sum(probs)
				move = np.random.choice(
					acts,
					p=probs)
				self.mcts.update_with_move(move)
			else:
				move = np.random.choice(acts, p=probs)
				self.mcts.update_with_move(move)
			
			if return_prob:
				return move, move_probs
			else:
				return move
		else:
			print('Warning: board is full')
			