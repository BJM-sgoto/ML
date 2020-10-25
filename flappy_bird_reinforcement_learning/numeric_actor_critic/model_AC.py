#TODO : 1 output for action
# multiple instances
# check actor loss of https://shalabhsingh.github.io/Deep-RL-Flappy-Bird/

import random
import numpy as np
import tensorflow.compat.v1 as tf
from env import QuickEnvironment

tf.disable_v2_behavior()
tf.reset_default_graph()
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

GAMMA = 0.99
NUM_GAMES = 16
NUM_INPUTS = 3
NUM_ACTIONS = 2
BATCH_SIZE = 50
NUM_EPISODE = 10000

LR = 1e-4

class Model_AC:
	def __init__(self, sprite_folder='./images/'):
		self.reuse_actor = False
		self.reuse_critic = False
		self.env = QuickEnvironment(sprite_folder)
	
	def act(self, state_holder):
		with tf.variable_scope('actor', reuse=self.reuse_actor):
			output_holder = state_holder
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.elu)
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.elu)
			output_holder = tf.layers.dense(
				output_holder,
				units=1,
				activation=tf.nn.sigmoid)
			output_holder = tf.squeeze(output_holder, axis=1)
			return output_holder
		
	def critize(self, state_holder):
		with tf.variable_scope('critic', reuse=self.reuse_critic):
			output_holder = state_holder
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.elu)
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.elu)
			output_holder = tf.layers.dense(
				output_holder,
				units=1)
			output_holder = tf.squeeze(output_holder, axis=1)
			return output_holder
	
	def compute_actor_cost(self, action_prob_holder, action_holder, target_value_holder):
		cost = -tf.math.log(tf.reduce_sum(action_prob_holder * action_holder, axis=1) + 1e-8) * target_value_holder
		cost = tf.reduce_mean(cost)
		return cost
		
	def compute_critic_cost(self, value_holder, target_value_holder):
		cost = tf.square(value_holder - target_value_holder)
		cost = tf.reduce_mean(cost)
		return cost
		
	def compute_entropy(self, action_prob_holder):
		entropy = tf.math.log(action_prob_holder + 1e-8) * action_prob_holder
		entropy = -tf.reduce_mean(entropy)
		return entropy
		
	def choose_action(self, state, random_rate=0):
		var_val, action_prob = self.session.run([self.first_var, self.P_AP], feed_dict={self.S: np.float32([state])})
		action_prob = action_prob[0]
		action = 1
		if np.random.rand()>action_prob:
			action = 0
		return action
	
	def train_one(self, state, action, reward, next_state):
		_state = np.float32([state])
		value = self.session.run(self.P_V, feed_dict={self.S: _state})
		if next_state is None:
			_next_state = np.zeros([1, NUM_INPUTS], dtype=np.float32)
		else:
			_next_state = np.float32([next_state])
		next_value = self.session.run(self.P_V, feed_dict={self.S: _next_state})
		if next_state is None:
			target_value = np.float32([reward])
		else:
			target_value = reward + GAMMA * next_value
		delta_value = target_value - value
		action_choice = np.zeros([1, NUM_ACTIONS], dtype=np.float32)
		action_choice[0, action] = 1.0
		cost_val, _ = self.session.run([self.cost, self.train_op], feed_dict={
			self.S: _state, 
			self.A: action_choice,
			self.V: target_value,
			self.D: delta_value})
		return cost_val
		
	def train(self, model_path='./model/model_AC', resume=False):
		self.S = tf.placeholder(tf.float32, [None, NUM_INPUTS])
		self.A = tf.placeholder(tf.float32, [None, NUM_ACTIONS])
		self.D = tf.placeholder(tf.float32, [None])
		self.V = tf.placeholder(tf.float32, [None])
		
		self.P_AP = self.act(self.S)
		self.P_V = self.critize(self.S)
		self.first_var = tf.trainable_variables()[0]
		
		actor_cost = self.compute_actor_cost(self.P_AP, self.A, self.D)
		critic_cost = self.compute_critic_cost(self.P_V, self.V)
		entropy = self.compute_entropy(self.P_AP)
		self.cost = actor_cost + 0.5 * critic_cost - 0.1*entropy
		#rint(self.cost, [actor_cost, critic_cost, entropy], 'Loss')
		self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
				
		saver = tf.train.Saver()
		self.session = tf.Session()
		if resume:
			saver.restore(self.session, model_path)
		else:
			self.session.run(tf.global_variables_initializer())
			
		for i in range(NUM_EPISODE):
			state = self.env.reset()
			duration = 0
			done = False
			sum_cost = 0
			while not done:
				action = self.choose_action(state)
				next_state, reward, done = self.env.step(action)
				if done:
					next_state = None
				duration += 1
				cost_val	= self.train_one(state, action, reward, next_state)
				sum_cost += cost_val
				state = next_state
			print('Episode {:04d} Duration{:03d} Loss {:05f}'.format(i, duration, sum_cost/duration))
		saver.save(self.session, model_path)
		self.session.close()
		
model = Model_AC(sprite_folder='./images/')
model.train(
	model_path='./model/model_AC', 
	resume=False)