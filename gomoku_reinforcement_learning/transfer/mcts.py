# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import numpy as np
from mcts_pure import MCTS as BaseMCTS, MCTSPlayer as BaseMCTSPlayer
from constants import P_E, NUM_INSTANCE, VIRTUAL_LOSS

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs = probs / np.sum(probs)
	return probs
	

class MCTS(BaseMCTS):
	def __init__(self, policy_value_fn, n_playout=10000):
		super(MCTS, self).__init__(policy_value_fn, n_playout)
		self._policy = policy_value_fn
		self._n_playout = n_playout // NUM_INSTANCE
	
	def _playout(self, board):
		# find the leaf node
		node = self._root
		actions = []
		nodes, actions_s = node.select_multiple(num=NUM_INSTANCE, virtual_loss=VIRTUAL_LOSS)
		
		# store original state
		ori_state = board.export_state()		
		state_batch = []
		ends = [None for i in range(NUM_INSTANCE)]
		leaf_values = np.zeros([NUM_INSTANCE], dtype=np.float32)
		availables_s = []
		for i in range(NUM_INSTANCE):
			actions = actions_s[i]
			for action in actions:
				board.do_move(action)
			
			state_batch.append(board.current_state())
			availables_s.append(board.availables.copy())
			end, winner = board.get_result()
			ends[i] = end
			if not end:
				leaf_value = np.NAN
			else:
				if winner == P_E: # tie
					leaf_value = 0.0
				else:
					leaf_value = 1.0 if winner == board.current_player else -1.0
			leaf_values[i] = leaf_value
			
			# restore original state
			board.import_state(ori_state)
			
		state_batch = np.int8(state_batch)
		action_probs_batch, leaf_value_batch = self._policy(state_batch)
		for i in range(NUM_INSTANCE):
			node = nodes[i]
			if not ends[i]:
				# we might select the same node several times, we only expand once
				# expand function only works if the node is truly leaf node
				node.expand(availables_s[i], action_probs_batch[i][availables_s[i]])
				node.backpropagate(-leaf_value_batch[i], virtual_loss=VIRTUAL_LOSS)
			else:
				node.backpropagate(-leaf_values[i], virtual_loss=VIRTUAL_LOSS)
		
	# use deterministic action
	def get_move_probs(self, board, temp=1e-3):
		for n in range(self._n_playout):
			self._playout(board)
		acts = self._root._child_ids
		visits = self._root._child_n_visits
		print('visits', visits)
		act_probs = softmax(np.log(np.array(visits) + 1e-10)/temp)
		return acts, act_probs
		
class MCTSPlayer(BaseMCTSPlayer):
	def __init__(self, policy_value_fn, n_playout=2000, is_selfplay=False):
		self.mcts  = MCTS(policy_value_fn, n_playout)
		self._is_selfplay = is_selfplay
		
	def get_action(self, board, temp=1e-3, return_prob=False):
		sensible_moves = board.availables
		move_probs = np.zeros(board.width * board.height, dtype=np.float32)
		if len(sensible_moves)>0:	
			acts, probs = self.mcts.get_move_probs(board, temp)
			move_probs[list(acts)] = probs
			if self._is_selfplay:
				probs = 0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
				probs = probs/ np.sum(probs)
				move = np.random.choice(
					acts,
					p = probs)
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