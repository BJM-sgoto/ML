# inference : https://github.com/junxiaosong/AlphaZero_Gomoku

import tensorflow.compat.v1 as tf
import numpy as np
from constants import RANDOM_SEED, P_E, DECAY

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)

class PolicyValueNet:	
	def __init__(self, board_width, board_height, model_file=None):
		self.board_width = board_width
		self.board_height = board_height
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
		
		# optimizer
		self.learning_rate = tf.placeholder(tf.float32)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		if model_file is not None:
			self.restore_model(model_file)
			
	def policy_value_op(self, input_state):
		conv1 = tf.layers.conv2d(
			inputs=input_state,
			filters=32,
			kernel_size=[3,3],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(
			inputs=conv1,
			filters=64,
			kernel_size=[3,3],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
		conv3 = tf.layers.conv2d(
			inputs=conv2,
			filters=128,
			kernel_size=[3,3],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
			
		# action network
		valid_actions = tf.cast(tf.equal(input_state, P_E), dtype=tf.float32)
		valid_actions = tf.layers.flatten(valid_actions)
		action_conv = tf.layers.conv2d(
			inputs=conv3,
			filters=4,
			kernel_size=[1,1],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
		action_conv_flat = tf.layers.flatten(action_conv)		
		action_fc = tf.layers.dense(
			inputs=action_conv_flat,
			units=self.board_width*self.board_height)
		policy = action_fc - tf.reduce_max(action_fc, axis=1, keepdims=True)
		policy = tf.exp(policy) * valid_actions
		policy = policy / tf.reduce_sum(policy, axis=1, keepdims=True)
		policy = tf.log(policy + 1e-10)
		print('Policy', policy)
		
		# evaluation network
		evaluation_conv = tf.layers.conv2d(
			inputs=conv3,
			filters=2,
			kernel_size=[1,1],
			padding='same',
			data_format='channels_last',
			activation=tf.nn.relu)
		evaluation_conv_flat = tf.layers.flatten(evaluation_conv)
		evaluation_fc1 = tf.layers.dense(
			inputs=evaluation_conv_flat,
			units=64,
			activation=tf.nn.relu)
		evaluation_fc2 = tf.layers.dense(
			inputs=evaluation_fc1,
			units=1,
			activation=tf.nn.tanh)
		return policy, evaluation_fc2
	
	def policy_value(self, state_batch):
		log_act_probs, value = self.session.run(
			[self.policy, self.evaluation],
			feed_dict={self.input_states:state_batch})
		act_probs = np.exp(log_act_probs)
		return act_probs, value
		
	def policy_value_fn(self, board):
		legal_positions = board.availables
		current_state = board.current_state().reshape((-1,self.board_height, self.board_width))
		act_probs, value = self.policy_value(current_state)
		act_probs = zip(legal_positions, act_probs[0][legal_positions])
		value = value[0, 0]
		return act_probs, value
		
	def train_step(self, state_batch, mcts_probs, winner_batch, lr):
		batch_size = len(winner_batch)
		target_value = winner_batch.copy()
		for i in range(batch_size):
			target_value[i] = target_value[i] * (DECAY** (batch_size - 1 - i))
		winner_batch = np.reshape(winner_batch, (-1,1))
		loss, entropy, _ = self.session.run(
			[self.loss, self.entropy, self.optimizer],
			feed_dict={
				self.input_states: state_batch,
				self.mcts_probs: mcts_probs,
				self.labels: target_value,
				self.learning_rate: lr})
		return loss, entropy
	
	def save_model(self, model_path):
		self.saver.save(self.session, model_path)
		
	def restore_model(self, model_path):
		self.saver.restore(self.session, model_path)