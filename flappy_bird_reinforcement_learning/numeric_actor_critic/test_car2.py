import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np
import gym  #requires OpenAI gym installed
import sklearn.preprocessing
import random

env = gym.envs.make("MountainCarContinuous-v0") 

tf.disable_v2_behavior()
tf.reset_default_graph()


NUM_EPISODES = 300
GAMMA = 0.99

lr_actor = 2e-5
lr_critic = 1e-3

class Memory:
	def __init__(self, max_memory=4000):
		self.max_memory = max_memory
		self.memory = []

	# sample format :  [episode_id, state, action, reward, next_state, evaluation]
	def add_sample(self, sample):
		self.memory.append(sample)
		if len(self.memory)>self.max_memory:
			self.memory.pop(0)

	def sample(self, n_samples):
		if n_samples > len(self.memory):
			return random.sample(self.memory, len(self.memory))
		else:
			return random.sample(self.memory, n_samples)

class Model:
	def __init__(self):
		self.env = gym.envs.make("MountainCarContinuous-v0") 
		self.n_inputs = len(self.env.observation_space.sample())
		self.n_outputs = len(self.env.action_space.sample())
		self.reuse_actor = False
		self.reuse_critic = False
		self.memory = Memory()
		
	def act(self, state_holder, training=False):
		with tf.variable_scope('actor', reuse=self.reuse_actor):
			
			state_holder = tf.layers.batch_normalization(
				state_holder, 
				training=training)
			
			state_holder = tf.layers.dense(
				state_holder,
				units=400,
				activation=tf.nn.elu)
			state_holder = tf.layers.dense(
				state_holder,
				units=400,
				activation=tf.nn.elu)
			mean_holder = tf.layers.dense(
				state_holder,
				units=self.n_outputs)
			sigma_holder = tf.layers.dense(
				state_holder,
				units=self.n_outputs)
			sigma_holder = tf.nn.softplus(sigma_holder) + 1e-5
			sampler = tfp.distributions.Normal(loc=mean_holder, scale=sigma_holder)
			sample = sampler.sample()
			action_holder = tf.clip_by_value(sample, self.env.action_space.low[0], self.env.action_space.high[0])
			self.reuse_actor = True
			return action_holder, sampler
			
	def critize(self, state_holder, training=False):
		with tf.variable_scope('critic', reuse=self.reuse_critic):
			
			state_holder = tf.layers.batch_normalization(
				state_holder, 
				training=training)
			
			value_holder = tf.layers.dense(
				state_holder,
				units=40,
				activation=tf.nn.elu)
			value_holder = tf.layers.dense(
				value_holder,
				units=40,
				activation=tf.nn.elu)
			value_holder = tf.layers.dense(
				value_holder,
				units=self.n_outputs)
			self.reuse_critic = True
			return value_holder
			
	def compute_actor_cost(self, sampler, action_holder, delta_holder):
		cost = -tf.math.log(sampler.prob(action_holder) + 1e-5) * delta_holder
		cost = tf.reduce_mean(cost)
		return cost
		
	def compute_critic_cost(self, value_holder, target_value_holder):
		cost = tf.square(value_holder - target_value_holder)
		cost = tf.reduce_mean(cost)
		return cost
	
	def scale_state(self, state):                 #requires input shape=(2,)
		scaled = self.scaler.transform([state])[0]
		return scaled 
		
	def train_batch(self):
		samples = self.memory.sample(50)
		states = np.float32([sample[0] for sample in samples])
		rewards = np.float32([[sample[1]] for sample in samples])
		dones = [sample[2] for sample in samples]
		actions = np.float32([sample[3] for sample in samples])
		next_states = np.float32([sample[4] if not sample[4] is None else np.zeros([self.n_inputs]) for sample in samples])
		
		value = self.session.run(self.P_V, feed_dict={self.X: states})
		next_value = self.session.run(self.P_V, feed_dict={self.X: next_states})
		target_value = GAMMA * next_value + rewards
		td_delta = target_value - value
		
		actor_cost_val, _ = self.session.run([self.actor_cost, self.actor_train_op], feed_dict={self.X: states, self.D: td_delta, self.A: actions})
		critic_cost_val, _ = self.session.run([self.critic_cost, self.critic_train_op], feed_dict={self.X: states, self.V: target_value})
		return actor_cost_val, critic_cost_val
	
	def train(self, model_path='./model/model', resume=False):
		state_space_samples = np.float32([self.env.observation_space.sample() for x in range(10000)])
		self.scaler = sklearn.preprocessing.StandardScaler()
		self.scaler.fit(state_space_samples)
		
		self.X = tf.placeholder(tf.float32, [None, self.n_inputs]) # input holder
		self.D = tf.placeholder(tf.float32, [None, self.n_outputs]) # delta holder
		self.V = tf.placeholder(tf.float32, [None, self.n_outputs]) # value holder
		self.A = tf.placeholder(tf.float32, [None, self.n_outputs]) # action holder
		self.P_A, self.sampler = self.act(self.X, training=True)
		self.P_V = self.critize(self.X, training=True)
		self.actor_cost = self.compute_actor_cost(self.sampler, self.A, self.D)
		self.critic_cost = self.compute_critic_cost(self.P_V, self.V)
		self.actor_train_op = tf.train.AdamOptimizer(lr_actor).minimize(self.actor_cost, var_list=tf.trainable_variables('actor'))
		self.critic_train_op = tf.train.AdamOptimizer(lr_critic).minimize(self.critic_cost, var_list=tf.trainable_variables('critic'))
				
		self.saver = tf.train.Saver()
		self.session = tf.Session()
		if resume:
			self.saver.restore(self.session, model_path)
		else:
			self.session.run(tf.global_variables_initializer())
		
		for i in range(NUM_EPISODES):
			done = False
			state = self.env.reset()
			duration = 0
			cum_reward = 0
			cum_critic_loss = 0
			cum_actor_loss = 0
			while not done:
				state = self.scale_state(state)
				#state = np.float32([state])
				action = self.session.run(self.P_A, feed_dict={self.X: np.float32([state])})[0]
				next_state, reward, done, _ = self.env.step(action)
				if next_state[0]<0.4:
					reward = 0
				else:
					reward = 10
				if done:
					next_state = None
				
				self.memory.add_sample([state, reward, done, action, next_state])
				actor_cost_val, critic_cost_val = self.train_batch()
				cum_critic_loss += critic_cost_val
				cum_actor_loss += actor_cost_val
				duration += 1
				
				cum_reward += reward
				state = next_state
			cum_actor_loss/=duration
			cum_critic_loss/=duration
			print('Episode {:03d} Duration {:03d} Actor Loss {:05f} Critic Loss {:05f} Reward {:06f}'.format(i, duration, cum_actor_loss, cum_critic_loss, cum_reward))
		#saver.save(session, model_path)
		self.session.close()
		
model = Model()
model.train(
	model_path='./model/model', 
	resume=False)
		