# this code to run on CPU
# generate 1 episode in 1 turn 

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
from env_rotation import Environment, REWARD_NOT_SOLVED, REWARD_SOLVED
from mcts import MCTS

RANDOM_SEED = 1237
BOARD_SIZE = 4

#tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Model:
	def __init__(self):
		self.env = Environment(height=BOARD_SIZE, width=BOARD_SIZE)
		self.reuse = False
		self.X = tf.placeholder(tf.float32, shape=[None] + self.env.state_size)
		self.P_P, self.P_V = self.init_model(self.X)
		
	def make_episode(self):
		actions = []		
		self.env.reset(reset_to_solved_state=True)
		
		for j in range(self.env.n_actions): # shuffle n_actions times
			action = self.env.sample_action()
			self.env.step(action)
			actions.insert(0, self.env.invert_action(action))
		init_state = self.env.export_state()
		return init_state, actions
	
	def init_model(self, state):
		with tf.variable_scope('model', reuse=self.reuse):
			feature = tf.layers.flatten(state)
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
				units=self.env.n_actions)
			policy = policy - tf.reduce_max(policy, axis=1, keepdims=True)
			policy = tf.exp(policy)
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
		init_state, actions = self.make_episode()
		
		states = []
		next_values = []
		next_policies = []
		self.env.import_state(init_state)
		
		n_actions = len(actions)
		rewards = [np.zeros([self.env.n_actions], dtype=np.float32) for i in range(n_actions)]
		#loss_weights = 1.0 / np.arange(n_actions, 0, -1, dtype=np.float32)
		loss_weights = 1.0/np.flip(np.arange(n_actions, dtype=np.float32)/(n_actions/4)+1, axis=0)
		
		for i in range(n_actions):
			# get current state
			states.append(self.env.get_state())
			action = actions[i]
			
			# check all next states
			next_states = []
			rewards = []
			for t_action in self.env.actions:
				reward = self.env.step(t_action)
				rewards.append(reward)
				next_states.append(self.env.get_state())
				self.env.undo(t_action)
			# move
			self.env.step(action)
			# next value
			t_next_value = session.run(self.P_V, feed_dict={self.X: next_states})
			for i, reward in enumerate(rewards):
				if reward!=REWARD_SOLVED:
					t_next_value[i]+=reward
				else:
					t_next_value[i] = REWARD_SOLVED
			next_value = np.max(t_next_value)
			next_values.append(next_value)
			pos_max = np.argmax(t_next_value)
			#pos_max = valid_actions[pos_max]# TODO check this
			next_policy = np.zeros([self.env.n_actions], dtype=np.float32)
			next_policy[pos_max] = 1.0
			next_policies.append(next_policy)
			
		states = np.float32(states)
		target_values = np.float32(next_values)
		target_policies = np.float32(next_policies)
		
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
		policy_loss = np.sum(policy_loss)/ np.sum(loss_weights)
		evaluation_loss = np.sum(evaluation_loss)/ np.sum(loss_weights)
		#print('Target Values', values)
		#exit()
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
		self.T_P = tf.placeholder(tf.float32, [None, self.env.n_actions])
		self.T_V = tf.placeholder(tf.float32, [None])
		self.policy_loss = tf.reduce_mean(tf.square(self.T_P - self.P_P), axis=1)
		self.evaluation_loss = tf.square(self.T_V - self.P_V)
		
		self.loss_weight = tf.placeholder(tf.float32, [None])
		
		self.loss = (self.policy_loss + self.evaluation_loss) * self.loss_weight
		self.loss = tf.reduce_sum(self.loss)/ tf.reduce_sum(self.loss_weight)
		
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
		return policy, value
	
			
	def naive_test(self, model_path='model/model'):
		self.session = tf.Session()
		self.saver = tf.train.Saver()
		self.saver.restore(self.session, model_path)
		
		self.env.reset(reset_to_solved_state=False,n_step=7)
		self.env.render()
		for i in range(10):
			if self.env.get_result()== REWARD_SOLVED:
				break
			policy, value = self.policy_value_fn(self.env)
			action = np.argmax(policy)
			self.env.step(action)
			print('Time', i, 'Action', action, 'Policy', policy)
			self.env.render()
		self.session.close()
	
	def test(self, model_path='model/model'):
		self.session = tf.Session()
		self.saver = tf.train.Saver()
		self.saver.restore(self.session, model_path)
	
		search_tree = MCTS(policy_value_fn=self.policy_value_fn, c_puct=5.0, n_playout=5000)
		self.env.reset(reset_to_solved_state=False,n_step=7)
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
	#model.train(num_epoch=50000, cache_file='./model/cache.txt', model_path='./model/model', resume=False)
	
	# naive method can solve 3-step shuffle
	#model.naive_test(model_path='./model/model')
	
	model.test(model_path='./model/model')