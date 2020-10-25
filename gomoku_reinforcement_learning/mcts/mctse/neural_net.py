# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import tensorflow.compat.v1 as tf
import numpy as np
from constants import RANDOM_SEED, P_E, BOARD_WIDTH, BOARD_HEIGHT, N_IN_ROW

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)

class PolicyValueNet:	
	def __init__(self, session, model_name, board_width, board_height, model_file=None):
		self.board_width = board_width
		self.board_height = board_height
		self.session = session
		self.model_name = model_name
		with tf.variable_scope(self.model_name):
			self.input_states = tf.placeholder(tf.int8, shape=[None, board_height, board_width])
			input_state = tf.expand_dims(tf.cast(self.input_states, dtype=tf.float32), -1)
			self.policy, self.evaluation = self.policy_value_op(input_state)
				
			# loss
			# value loss
			self.labels = tf.placeholder(tf.float32, [None, 1])
			self.value_loss = tf.losses.mean_squared_error(self.labels, self.evaluation)
			
			# policy loss
			self.mcts_probs = tf.placeholder(tf.float32, [None, board_width*board_height])
			self.policy_loss = -tf.reduce_mean(tf.reduce_sum(self.mcts_probs * self.policy, axis=1))
			
			# penalty loss
			vars = tf.trainable_variables()
			l2_penalty = 1e-4 * tf.add_n([tf.nn.l2_loss(var) for var in vars if 'bias' not in var.name.lower()]) #??
			
			self.loss = self.value_loss + self.policy_loss + l2_penalty
			
			# entropy
			self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.policy) * self.policy, 1)))
			
			# learning rate
			self.learning_rate = tf.placeholder(tf.float32)
			
			# optimizer
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=tf.trainable_variables(model_name))
			self.saver = tf.train.Saver(var_list=tf.global_variables(model_name))
		if model_file is not None:
			self.restore_model(model_file)
			
	def policy_value_op(self, input_state):
		conv = tf.layers.conv2d(
			inputs=input_state,
			filters=64,
			kernel_size=[3,3],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
		n_layer = N_IN_ROW - 1
		for i in range(n_layer):
			conv = tf.layers.conv2d(
				inputs=conv,
				filters=64,
				kernel_size=[3,3],
				padding='same',
				data_format='channels_last',
				activation=tf.nn.relu)
			
		# action network
		valid_actions = tf.cast(tf.equal(input_state, P_E), dtype=tf.float32)
		valid_actions = tf.layers.flatten(valid_actions)
		policy = tf.layers.conv2d(
			inputs=conv,
			filters=4,
			kernel_size=[1,1],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
		policy = tf.layers.flatten(policy)		
		policy = tf.layers.dense(
			inputs=policy,
			units=self.board_width*self.board_height)
		policy = policy - tf.reduce_max(policy, axis=1, keepdims=True)
		policy = tf.exp(policy) * valid_actions
		policy = policy / tf.reduce_sum(policy, axis=1, keepdims=True)
		policy = tf.log(policy + 1e-10)
		
		# evaluation network
		evaluation = tf.layers.conv2d(
			inputs=conv,
			filters=2,
			kernel_size=[1,1],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
		evaluation = tf.layers.flatten(evaluation)
		evaluation = tf.layers.dense(
			inputs=evaluation,
			units=64,
			activation=tf.nn.relu)
		evaluation = tf.layers.dense(
			inputs=evaluation,
			units=1,
			activation=tf.nn.tanh)
		return policy, evaluation
	
	def policy_value_fn(self, state_batch):
		log_act_probs, value = self.session.run(
			[self.policy, self.evaluation],
			feed_dict={self.input_states:state_batch})
		act_probs = np.exp(log_act_probs)
		return act_probs, value
	
	'''	
	def policy_value_fn(self, board):
		legal_positions = board.availables
		current_state = board.current_state().reshape((-1,self.board_height, self.board_width))
		act_probs, value = self.policy_value(current_state)
		act_probs = zip(legal_positions, act_probs[0][legal_positions])
		value = value[0, 0]
		return act_probs, value
	'''
	
	def train_step(self, state_batch, mcts_probs, winner_batch, lr):
		winner_batch = np.reshape(winner_batch, (-1,1))
		loss, entropy, _ = self.session.run(
			[self.loss, self.entropy, self.optimizer],
			feed_dict={
				self.input_states: state_batch,
				self.mcts_probs: mcts_probs,
				self.labels: winner_batch,
				self.learning_rate: lr})
		return loss, entropy
	
	def save_model(self, model_path):
		self.saver.save(self.session, model_path)
		
	def restore_model(self, model_path):
		self.saver.restore(self.session, model_path)