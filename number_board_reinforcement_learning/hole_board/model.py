# this code should be run on GPU, because it takes time to process raw states
# generate 1 episode in 1 turn 

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
from env_hole_board import Environment, REWARD_NOT_SOLVED, REWARD_SOLVED, REWARD_UNCHANGED
from mcts import MCTS

RANDOM_SEED = None
BOARD_SIZE = 4

#tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Model:
	def __init__(self):
		self.env = Environment(BOARD_SIZE, BOARD_SIZE)
		self.reuse = False
		self.X = tf.placeholder(tf.int32, shape=[None] + self.env.state_size)
		self._lookup_table = self._make_lookup_table(BOARD_SIZE, BOARD_SIZE)
		self.P_P, self.P_V = self.init_model(self.X)
		#print(self.P_P, self.P_V)
		
	def _make_lookup_table(self, height, width):
		num_pieces = height * width
		piece_size = height * width
		lookup_table = np.zeros([num_pieces, piece_size], dtype=np.float32)
		for i in range(num_pieces):
			lookup_table[i,i] = 1.0
		return lookup_table
		
	def make_episodes(self, n_episodes=10):
		actions_s = []
		init_states = []
		for i in range(n_episodes):
			actions = []		
			self.env.reset(reset_to_solved_state=True)
			for j in range(20): # shuffle 20 times
				action = self.env.sample_action()
				reward = self.env.step(action)
				# only generate meaningful actions
				if reward == REWARD_UNCHANGED:
					continue
				action = self.env.invert_action(action)
				actions.insert(0, action)
			init_states.append(self.env.export_state())
			actions_s.append(actions)
		return init_states, actions_s
	
	# TODO:  state is raw board
	def init_model(self, state):
		valid_positions = np.ones([BOARD_SIZE, BOARD_SIZE, 4], dtype=np.float32)
		valid_positions[:, 0, Environment.LEFT] = 0.0
		valid_positions[0, :, Environment.UP] = 0.0
		valid_positions[:, -1, Environment.RIGHT] = 0.0
		valid_positions[-1, :, Environment.DOWN] = 0.0
		valid_positions = valid_positions.flatten()
		
		# cell action lookup table
		cell_action_lookup_table = np.float32([[0,0,0,0],[1,1,1,1]])
		n_actions = valid_positions.shape[0]
		with tf.variable_scope('model', reuse=self.reuse):
			feature = tf.reshape(state, [-1, BOARD_SIZE * BOARD_SIZE, 1]) # shape : [None, BOARD_SIZE * BOARD_SIZE, 1]
			hole_pos = tf.where(tf.equal(feature, 0), tf.ones_like(feature), tf.zeros_like(feature)) # shape : [None, BOARD_SIZE * BOARD_SIZE, 1]
			cell_actions = tf.gather_nd(cell_action_lookup_table, hole_pos) # shape : [None, BOARD_SIZE * BOARD_SIZE, 4]
			cell_actions = tf.layers.flatten(cell_actions)
			feature = tf.gather_nd(self._lookup_table, feature) # shape : [None, BOARD_SIZE * BOARD_SIZE, PIECE_SIZE]
			feature = tf.layers.flatten(feature)
			feature = tf.layers.dense(
				feature,
				units=1024,
				activation=tf.nn.elu)
			feature = tf.layers.dense(
				feature,
				units=512,
				activation=tf.nn.elu)
				
			# policy net	
			policy = tf.layers.dense(
				feature,
				units=128)
			policy = tf.layers.dense(
				policy,
				units=n_actions)
			policy = policy - tf.reduce_max(policy, axis=1, keepdims=True)
			policy = tf.exp(policy) * valid_positions * cell_actions
			policy = policy / tf.reduce_sum(policy, axis=1, keepdims=True)
				
			# evaluation net
			evaluation = tf.layers.dense(
				feature,
				units=128)
			evaluation = tf.layers.dense(
				evaluation,
				units=1,
				activation=tf.nn.tanh)
			evaluation = tf.squeeze(evaluation, axis=1)
			self.reuse = True
			return policy, evaluation
			
	def train_episode(self, session):
		n_episodes = 1
		init_states, actions_s = self.make_episodes(n_episodes=n_episodes)
		
		states = []
		next_states = []
		
		rewards_s = []
		valid_actions_s = []
		valid_action_counts = []
		
		loss_weights = []
		hole_poss = []
		
		for i in range(n_episodes):
			init_state = init_states[i]
			actions = actions_s[i]
			self.env.import_state(init_state)
			n_actions = len(actions)
			
			loss_weight = 1.0 / np.arange(n_actions, 0, -1, dtype=np.float32)
			loss_weights.append(loss_weight)
				
			for j in range(n_actions):
				states.append(self.env.get_state())
				valid_actions = self.env.get_valid_actions()
				valid_actions_s.append(valid_actions)
				valid_action_counts.append(len(valid_actions))
				
				rewards = []
				for t_action in valid_actions:
					reward = self.env.step(t_action)
					rewards.append(reward)
					next_states.append(self.env.get_state())
					self.env.undo(t_action)
				
				hole_pos = self.env.hole_pos
				hole_poss.append(hole_pos[0] * BOARD_SIZE + hole_pos[1])
				self.env.step(actions[j])
				rewards_s.append(rewards)
		
		states = np.float32(states)
		next_states = np.float32(next_states)
		loss_weights = np.concatenate(loss_weights)
		# compute values , next values
		values = session.run(self.P_V, feed_dict={
			self.X : states})
		next_values = session.run(self.P_V, feed_dict={
			self.X : next_states})
		
		# make target policy and target values
		n = values.shape[0]
		target_values = np.zeros([n], dtype=np.float32)
		target_policies = np.zeros([n, BOARD_SIZE * BOARD_SIZE * 4], dtype=np.float32)
		for i in range(n):
			valid_actions = valid_actions_s[i]
			rewards = rewards_s[i]
			for j, valid_action in enumerate(valid_actions):
				if rewards[j] == REWARD_SOLVED:
					target_values[i] = REWARD_SOLVED
					target_policies[i, hole_poss[i] * 4 + valid_action] = 1.0
					break
			else:
				start_ep = int(np.sum(valid_action_counts[:i]))
				end_ep = int(np.sum(valid_action_counts[:i+1]))
				t_next_values = next_values[start_ep: end_ep]
				target_values[i] = REWARD_NOT_SOLVED + np.max(t_next_values)
				max_pos = np.argmax(t_next_values)
				best_action = hole_poss[i] * 4 + valid_actions[max_pos]
				target_policies[i, best_action] = 1.0
		
		# train
		values, policies, policy_loss, evaluation_loss, _ = session.run([
			self.P_V,
			self.P_P,
			self.policy_loss,
			self.evaluation_loss,
			self.train_op], 
			feed_dict={self.X: states,
				self.T_P: target_policies,
				self.T_V: target_values,
				self.loss_weight: loss_weights})
		'''
		print('hole poss')
		print(hole_poss)
		print('actions_s')
		print(actions_s)
		print('target_policies')
		print(target_policies)
		print('policies')
		print(policies)
		exit()
		'''
		###############
		return policy_loss, evaluation_loss
		
	def write_cache(self, cache_file, best_loss):
		f = open(cache_file, 'w')
		f.write(str(best_loss))
		f.close()
		
	def read_cache(self, cache_file):
		f = open(cache_file, 'r')
		best_loss = float(f.read().strip())
		f.close()
		return best_loss
	
	def train(self, num_epoch=10000, cache_file='./model/cache.txt', model_path='./model/model', resume=False):
		self.T_P = tf.placeholder(tf.float32, [None, BOARD_SIZE * BOARD_SIZE * 4])
		self.T_V = tf.placeholder(tf.float32, [None])
		self.policy_loss = tf.reduce_sum(tf.square(self.T_P - self.P_P), axis=1)
		self.evaluation_loss = tf.square(self.T_V - self.P_V)
		
		self.loss_weight = tf.placeholder(tf.float32, [None])
		
		self.loss = (self.policy_loss + self.evaluation_loss) * self.loss_weight
		self.loss = tf.reduce_mean(self.loss)
		
		self.train_op = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
		self.session = tf.Session()
		self.saver = tf.train.Saver()
		best_loss = 1.0
		if resume:
			self.saver.restore(self.session, model_path)
			best_loss = self.read_cache(cache_file)
		else:
			self.session.run(tf.global_variables_initializer())
			
		try:
			check_freq = 500
			average_loss = 0
			for i in range(num_epoch):
				policy_loss, evaluation_loss = self.train_episode(self.session)
				policy_loss = np.mean(policy_loss)
				evaluation_loss = np.mean(evaluation_loss)
				average_loss += policy_loss + evaluation_loss
				print('Epoch {:05d}, Policy Loss {:06f}, Evaluation Loss {:06f}, Best loss {:06f}'.format(i, policy_loss, evaluation_loss, best_loss))
				
				if (i+1)%check_freq==0:
					average_loss = average_loss / check_freq
					print('-----------------')
					print('Average Loss', average_loss)
					if average_loss < best_loss:
						print('Save model!')
						best_loss = average_loss
						self.saver.save(self.session, model_path)
						self.write_cache(cache_file, best_loss)
					print('-----------------')
					average_loss = 0
		except KeyboardInterrupt:
			print('Quit')
			self.session.close()
		else:
			print('Quit safely')
			self.session.close()
		
	def policy_value_fn(self, env):
		policy, value = self.session.run(
			[self.P_P, self.P_V], 
			feed_dict={
				self.X: np.int8([env.get_state()])})
		policy = policy[0]
		value = value[0]
		policy = np.exp(policy-np.max(policy))
		policy = policy/np.sum(policy)
		valid_actions = env.get_valid_actions()
		hole_pos = env.hole_pos
		hole_pos = hole_pos[0] * BOARD_SIZE + hole_pos[1]
		extended_valid_actions = [valid_action + hole_pos * 4 for valid_action in valid_actions]
		policy = {valid_actions[i]: policy[extended_valid_actions[i]] for i in range(len(valid_actions))}
		return policy, value
	
			
	def naive_test(self, model_path='model/model'):
		self.session = tf.Session()
		self.saver = tf.train.Saver()
		self.saver.restore(self.session, model_path)
		
		self.env.reset(reset_to_solved_state=False, n_step=6)
		self.env.render()
		for i in range(10):
			if self.env.get_result()== REWARD_SOLVED:
				break
			policy, value = self.policy_value_fn(self.env)
			action = max(policy.items(), key=lambda item: item[1])[0]
			self.env.step(action)
			print('Time', i, 'Action', action)
			self.env.render()
			
		self.session.close()
	
	def test(self, model_path='model/model'):
		self.session = tf.Session()
		self.saver = tf.train.Saver()
		self.saver.restore(self.session, model_path)
	
		search_tree = MCTS(policy_value_fn=self.policy_value_fn, c_puct=5.0, n_playout=5000)
		self.env.reset(reset_to_solved_state=False,n_step=20)
		self.env.render()
		for i in range(100):
			if self.env.get_result()== REWARD_SOLVED:
				break
			actions, found_path = search_tree.get_move(self.env)
			for action in actions:
				print('-----------------\nStep {} Action {}'.format(i, action))
				n_visits = [search_tree._root._children[child]._n_visits for child in search_tree._root._children]
				print('Children', n_visits)
				self.env.step(action)
				self.env.render()
			if found_path:
				break
			search_tree.update_with_move(action)
			
		self.session.close()
		
if __name__=='__main__':
	model = Model()
	model.train(num_epoch=50000, cache_file='./model/cache.txt', model_path='./model/model', resume=False)
	#model.naive_test(model_path='./model/model')
	#model.test(model_path='./model/model')