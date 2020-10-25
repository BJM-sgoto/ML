#reference : https://arxiv.org/pdf/1805.07470.pdf
# https://towardsdatascience.com/learning-to-solve-a-rubiks-cube-from-scratch-using-reinforcement-learning-381c3bac5476
#https://medium.com/datadriveninvestor/reinforcement-learning-to-solve-rubiks-cube-and-other-complex-problems-106424cf26ff

# Note MCTS is used to find the path, not to train

import tensorflow.compat.v1 as tf
import numpy as np 
from env_cube import Environment, REWARD_NOT_SOLVED, REWARD_SOLVED

tf.disable_v2_behavior()
tf.reset_default_graph()

class Model:
	def __init__(self):
		self.env = Environment()
		self.reuse = False
		self.X = tf.placeholder(tf.float32, shape=[None] + self.env.state_size)
		self.P_P, self.P_V = self.init_model(self.X)		
		print(self.P_P, self.P_V)
		
	 # dataset is saved under list of episodes
	 # episodes format [sides, actions]
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
				units=2048,
				activation=tf.nn.elu)
			feature = tf.layers.dense(
				feature,
				units=1024,
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
			
	def train_episode(self, session):
		n_episode = 20
		init_sides, actions = self.make_episodes(n_episode)
		states = []
		next_states = []
		rewards = []
		for n in range(n_episode):
			self.env.init_from_sides(init_sides[n])
			episode_actions = actions[n]
			epi_rewards = [[] for i in range(len(episode_actions))]
			for i in range(len(episode_actions)):
				action = episode_actions[i]
				states.append(self.env.get_state())
				for t_action in range(self.env.n_action):
					reward = self.env.step(t_action)
					epi_rewards[i].append(reward)
					next_states.append(self.env.get_state())
					self.env.undo(t_action)
				self.env.step(action)
			rewards += epi_rewards
		states = np.float32(states)
		next_states = np.float32(next_states)
		rewards = np.float32(rewards)
		
		next_values = session.run([self.P_V], feed_dict={self.X: next_states})
		next_values = np.reshape(next_values, (-1, self.env.n_action))
		next_values = next_values + rewards
		target_values = np.max(next_values, axis=1)
		
		max_policy_pos = np.argmax(next_values, axis=1)
		target_policies = np.zeros([len(states), self.env.n_action], dtype=np.float32)
		
		for i in range(len(states)):
			for j in range(self.env.n_action):
				if rewards[i,j]==REWARD_SOLVED:
					target_policies[i,j] = 1.0
					target_values[i] = REWARD_SOLVED
					break
			else:
				target_policies[i,max_policy_pos[i]] = 1.0
		
		#print('next_values', np.array2string(next_values, precision =2, suppress_small =True))
		# train
		values, policies, cost_val, _ = session.run([self.P_V, self.P_P, self.cost, self.train_op], feed_dict={self.X: states, self.T_P: target_policies, self.T_V: target_values})
		return cost_val
			
	def train(self, num_episode=10000, model_path='./model/model', resume=False):
		self.T_P = tf.placeholder(tf.float32, shape=[None, self.env.n_action])
		self.T_V = tf.placeholder(tf.float32, shape=[None])
		policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.T_P, logits=self.P_P)
		evaluation_loss = tf.square(self.P_V - self.T_V)
		self.cost =  policy_loss + evaluation_loss 
		batch_size = tf.shape(self.X)[0]
		cost_weight = 1.0/tf.range(batch_size, 0, -1, dtype=tf.float32)
		self.cost = tf.reduce_mean(self.cost * cost_weight)
		self.train_op = tf.train.RMSPropOptimizer(1e-4).minimize(self.cost)
		
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
			
		for i in range(num_episode):
			cost_val = self.train_episode(session)
			print('Episode {:05d}, Loss {:05f}'.format(i, cost_val))
			if (i+1)%1000==0:
				saver.save(session, model_path)
				print('Save model')
		session.close()
	
	def test(self, model_path='./model/model'):
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		
		self.env.reset()
		init_side, action = self.make_episodes(1)
		init_side = init_side[0]
		action = action[0]
		print('-----------------------\nAction', action)
		self.env.render()
		for i in range(20):			
			state = self.env.get_state()
			state = np.float32([state])
			predicted_policy = session.run(self.P_P, feed_dict={self.X: state})[0]
			predicted_action = np.argmax(predicted_policy)
			print('Step', i, 'policy', np.array2string(predicted_policy, precision=2),'action', predicted_action)
			reward = self.env.step(predicted_action)
			self.env.render()
			if reward==REWARD_SOLVED:
				break
		session.close()

model = Model()	
model.train(num_episode=10000, model_path='./model/model_3X3', resume=True)
#model.test(model_path='./model/model_2X2')
