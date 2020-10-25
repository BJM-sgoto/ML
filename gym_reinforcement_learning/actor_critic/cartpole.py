import numpy as np
import tensorflow.compat.v1 as tf
import gym

tf.disable_v2_behavior()
tf.reset_default_graph()

BATCH_SIZE = 10 # must train the model with BATCH_SIZE sequential samples 
NUM_UPDATES = 4000

NUM_INPUTS = 4
NUM_OUTPUTS = 2

COEFF_ACTOR = 1.00
COEFF_CRITIC = 0.50
COEFF_ENTROPY = 0.0001
LR = 1e-3

GAMMA = 0.99
seed = 1235
tf.set_random_seed(seed)
np.random.seed(seed)

class Model:
	def __init__(self):
		self.env = gym.make('CartPole-v0')

	def compute_action_value(self, state_holder):
		hidden1 = tf.layers.dense(
			state_holder,
			units=128,
			activation=tf.nn.relu)
		hidden1 = tf.layers.dense(
			hidden1,
			units=128,
			activation=tf.nn.relu)
					
		hidden2 = tf.layers.dense(
			state_holder,
			units=128,
			activation=tf.nn.relu)
		hidden2 = tf.layers.dense(
			hidden2,
			units=128,
			activation=tf.nn.relu)
			
		action_prob_holder = tf.layers.dense(
			hidden1,
			units=NUM_OUTPUTS)
		action_prob_holder = tf.nn.softmax(action_prob_holder, axis=-1)
		
		value_holder = tf.layers.dense(
			hidden2,
			units=1)
		value_holder = tf.squeeze(value_holder, axis=1)
		
		return action_prob_holder, value_holder
	
	def compute_actor_loss(self, action_prob_holder, action_holder, advantage_holder):
		loss = -tf.math.log(tf.reduce_sum(action_prob_holder * action_holder, axis=1) + 1e-10)
		loss = tf.reduce_mean(loss * advantage_holder)
		return loss
		
	def compute_critic_loss(self, value_holder, target_value_holder):
		loss = tf.reduce_mean(tf.square(value_holder - target_value_holder))
		return loss
		
	def compute_entropy_loss(self, action_prob_holder):
		loss = tf.reduce_mean(action_prob_holder * tf.math.log(action_prob_holder + 1e-10) + (1-action_prob_holder) * tf.math.log(1-action_prob_holder + 1e-10))
		return loss
		
	def train_batch(self, states, actions, rewards, dones, next_state, values):
		next_value = self.session.run(self.P_V, feed_dict={self.S: np.float32([next_state])})
		target_value = rewards.copy()
		for i in reversed(range(BATCH_SIZE)):
			target_value[i] = rewards[i] + GAMMA * next_value * (1 - dones[i])
			next_value = target_value[i]
		
		#advantage = target_value - values
		loss_val, _ = self.session.run([self.loss, self.train_op], feed_dict={self.S: states, self.A: actions, self.V: target_value, self.D: target_value})
		return loss_val
	
	def train(self, model_path='./model_pole/model', resume=False):
		self.S = tf.placeholder(tf.float32, [None, NUM_INPUTS])
		self.V =  tf.placeholder(tf.float32, [None])
		self.D =  tf.placeholder(tf.float32, [None])
		self.A =  tf.placeholder(tf.float32, [None, NUM_OUTPUTS])
		self.P_AP, self.P_V = self.compute_action_value(self.S)
		actor_loss = self.compute_actor_loss(self.P_AP, self.A, self.D)
		critic_loss = self.compute_critic_loss(self.P_V, self.V)
		entropy_loss = self.compute_entropy_loss(self.P_AP)
		self.loss = COEFF_ACTOR * actor_loss + COEFF_CRITIC * critic_loss + COEFF_ENTROPY * entropy_loss
		self.train_op = tf.train.RMSPropOptimizer(LR).minimize(self.loss)
		
		self.session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(self.session, model_path)
		else:
			self.session.run(tf.global_variables_initializer())
		
		states = np.zeros([BATCH_SIZE, NUM_INPUTS], dtype=np.float32)
		actions = np.zeros([BATCH_SIZE, NUM_OUTPUTS], dtype=np.float32)
		rewards = np.zeros([BATCH_SIZE], dtype=np.float32)
		dones = np.zeros([BATCH_SIZE], dtype=np.float32)
		
		values = np.zeros([BATCH_SIZE], dtype=np.float32)
		
		state = self.env.reset()
		duration = 0
		for i in range(NUM_UPDATES):
			for j in range(BATCH_SIZE):
				states[j] = state.copy()
				action_prob, value = self.session.run([self.P_AP, self.P_V], feed_dict={self.S: np.float32([state])})
				action = np.random.choice(NUM_OUTPUTS, p=action_prob[0])
				values[j] = value[0]
				
				if action==0:
					actions[j, 0] = 1.
					actions[j, 1] = 0.
				else:
					actions[j, 1] = 1.
					actions[j, 0] = 0.
				
				state, rewards[j], dones[j], _ = self.env.step(action)
				duration += 1
				if dones[j]:
					state = self.env.reset()
					print('Update {:04d} Duration {:04d}'.format(i, duration))
					saver.save(self.session, model_path)
					duration = 0
			
			loss_val = self.train_batch(states, actions, rewards, dones, state, values)
		self.session.close()
			
	def test(self, model_path='./model_pole/model'):
		self.S = tf.placeholder(tf.float32, [None, NUM_INPUTS])
		self.P_AP, self.P_V = self.compute_action_value(self.S)
		
		self.session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(self.session, model_path)
		
		for i in range(100):
			duration = 0
			state = self.env.reset()
			done = False
			while not done:
				duration += 1
				action_prob = self.session.run(self.P_AP, feed_dict={self.S: np.float32([state])})[0]
				action = 0
				if action_prob[1]>action_prob[0]:
					action = 1
				state, reward, done, _ = self.env.step(action)
			print('Duration', duration)
		self.session.close()
			
				
model = Model()
#model.train(model_path='./model_pole/model', resume=False)
model.test()