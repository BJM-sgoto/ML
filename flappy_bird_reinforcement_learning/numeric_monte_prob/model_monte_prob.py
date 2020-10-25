# TODO1: compute sum of probabilities on one trajectory???

# TODO2: compute sum of probabilities (of the last 40 samples) on multiple trajectories???


import numpy as np 
import tensorflow.compat.v1 as tf
from env import QuickEnvironment
import gym

tf.disable_v2_behavior()
tf.reset_default_graph()

NUM_EPISODES = 1000
NUM_GAMES = 20
NUM_INPUTS = 3
NUM_ACTIONS = 2
MAX_STATE_LEN = 1024

LEARNING_RATE = 1e-3

#hyper parameter
GAMMA = 0.5

class Model:
	def __init__(self, sprite_folder='./images/'):
		self.reuse_encoder = False
		self.X = tf.placeholder(tf.float32, [None, NUM_INPUTS]) # store state
		self.A = tf.placeholder(tf.float32, [None, NUM_ACTIONS]) # store action
		self.R = tf.placeholder(tf.float32, [None]) # store discounted reward
		self.PY = self.encode(self.X)
		self.cost = self.compute_cost(self.PY, self.A, self.R)
		self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		
		self.saver = tf.train.Saver()
		self.session = tf.Session()
		
		self.memory = np.zeros([NUM_GAMES, MAX_STATE_LEN, NUM_INPUTS], dtype=np.float32)
		
		self.env = []
		for i in range(NUM_GAMES):
			self.env.append(QuickEnvironment(sprite_folder))
		
	
	# compute probabilities of actions
	def encode(self, input_holder):
		output_holder = input_holder
		with tf.variable_scope('encoder', reuse=self.reuse_encoder):
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.leaky_relu,
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.leaky_relu,
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
			output_holder = tf.layers.dense(
				output_holder,
				units=NUM_ACTIONS,
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
			output_holder = tf.nn.softmax(output_holder, axis=1)
			self.reuse_encoder = True
		return output_holder
		
	# choose action for one state
	def choose_action(self, state):
		action_prob = self.session.run(self.PY, feed_dict={self.X: np.expand_dims(state, axis=0)})[0]
		action = np.random.choice(NUM_ACTIONS, p=action_prob)
		return action
	
	def compute_cost(self, action_scores, actions, discounted_rewards):
		cost = -tf.math.log(tf.reduce_sum(action_scores * actions, axis = 1))
		cost = tf.reduce_mean(cost * discounted_rewards)
		return cost
	
	# rewards : r_(t), r_(t+1), r_(t+2), r_(t+3)...., r_(T)
	def discount_rewards(self, rewards):
		n_rewards = rewards.shape[0]
		discounted_rewards = rewards.copy()
		for i in reversed(range(n_rewards - 1)):
			discounted_rewards[i] += GAMMA * discounted_rewards[i+1]
		#discounted_rewards = (discounted_rewards - np.mean(discounted_rewards))/ np.std(discounted_rewards)
		return discounted_rewards
	
	def train_batch(self, states, actions, discounted_rewards):
		loss_val, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X: states, self.A: actions, self.R: discounted_rewards})
		return loss_val
	
	def train(self, model_path='./model/model/', resume=False):
		if resume:
			self.saver.restore(self.session, model_path)
		else:
			self.session.run(tf.global_variables_initializer())
		alive = np.ones([NUM_GAMES], dtype=np.float32)	
		for episode_id in range(int(NUM_EPISODES/NUM_GAMES)):
			# clear old data
			states = []
			actions = []
			rewards = []
			
			# init episode
			done = False
			state = self.env.reset()
			duration = 0
			while not done:
				duration += 1
				action = self.choose_action(state)
				next_state, reward, done = self.env.step(action)
				
				# save to train
				states.append(state)
				rewards.append(reward)
				action_choice = np.zeros([NUM_ACTIONS], dtype=np.float32)
				action_choice[action] = 1
				actions.append(action_choice)
				
				# move to next state
				state = next_state
				
			states = np.float32(states)
			discounted_rewards = self.discount_rewards(np.float32(rewards))
			actions = np.float32(actions)
			
			loss_val = self.train_batch(states, actions, discounted_rewards)
			print('Episode {:04d}, Duration {:04d}, Loss {:05f}'.format(episode_id, duration, loss_val))
			#if (episode_id + 1)%50==0:
			#	self.saver.save(self.session, model_path)
		self.session.close()

model = Model(sprite_folder='./images/')
model.train('model_monte_prob/model', resume=False)
