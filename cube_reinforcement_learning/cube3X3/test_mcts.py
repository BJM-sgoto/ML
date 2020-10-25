import tensorflow.compat.v1 as tf
import numpy as np
from mcts import MCTS
from env_cube import REWARD_SOLVED, Environment

tf.disable_v2_behavior()
tf.reset_default_graph()

class TestModel:
	def __init__(self, model_path='./model/model'):
		self.env = Environment()
		self.reuse = False
		self.X = tf.placeholder(tf.float32, shape=[None] + self.env.state_size)
		self.P_P, self.P_V = self.init_model(self.X)
		self.session = tf.Session()
		self.saver = tf.train.Saver()
		self.saver.restore(self.session, model_path)
		
	def make_episodes(self, n_episode=20):
		init_sides = []
		actions = [[] for i in range(n_episode)]
		for i in range(n_episode):
			self.env.reset()
			# with 20 steps, we can reach any state of  a 3X3 cube
			for j in range(8): # small number of rotation for easy task
				action = self.env.sample_action()
				self.env.step(action)
				# revert action
				if action>=6:
					action-=6
				else:
					action+=6
				actions[i].insert(0, action)
			init_sides.append(self.env.get_sides())
		return init_sides, actions
		
	def init_model(self, state):
		with tf.variable_scope('model', reuse=self.reuse):
			feature = tf.layers.flatten(state)
			feature = tf.layers.dense(
				feature,
				units=4096,
				activation=tf.nn.elu)
			feature = tf.layers.dense(
				feature,
				units=2048,
				activation=tf.nn.elu)
				
			# policy net	
			policy = tf.layers.dense(
				feature,
				units=512,
				activation=tf.nn.elu)
			policy = tf.layers.dense(
				policy,
				units=12)
				
			# evaluation net
			evaluation = tf.layers.dense(
				feature,
				units=512,
				activation=tf.nn.elu)
			evaluation = tf.layers.dense(
				evaluation,
				units=1,
				activation=tf.nn.tanh)
			evaluation = tf.squeeze(evaluation, axis=1)
			self.reuse = True
			return policy, evaluation
	
	def policy_value_fn(self, cube):
		policy, value = self.session.run([self.P_P, self.P_V], feed_dict={self.X: np.float32([cube.get_state()])})
		policy = policy[0]
		value = value[0]
		policy = np.exp(policy - np.max(policy))
		policy = policy/ np.sum(policy)
		return zip(np.arange(self.env.n_action),policy), value
	
	def test(self):
		search_tree = MCTS(policy_value_fn=self.policy_value_fn, c_puct=5, n_playout=2000)
		self.env.reset()
		init_side, action = self.make_episodes(1)
		init_side = init_side[0]
		action = action[0]
		print('-----------------------\nAction', action)
		for i in range(100):
			self.env.render()
			if self.env.get_result()== REWARD_SOLVED:
				break
			actions, found_path = search_tree.get_move(self.env)
			for action in actions:
				print('-----------------\nStep {} Action {}'.format(i, action))
				self.env.step(action)
				self.env.render()
			if found_path:
				break
			search_tree.update_with_move(action)
			
		self.session.close()

test_model = TestModel(model_path='./model/model_3X3')
test_model.test()