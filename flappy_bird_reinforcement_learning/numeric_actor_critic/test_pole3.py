import gym
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

tf.disable_v2_behavior()
tf.reset_default_graph()

NUM_INPUTS = 4
NUM_ACTIONS = 2

class Model:
		
	def __call__(self, inputs):
		x = tf.convert_to_tensor(inputs, dtype=tf.float32)
		hidden_logs = tf.layers.dense(
			x,
			units=128,
			activation=tf.nn.relu)
		logits = tf.layers.dense(
			hidden_logs,
			units=NUM_ACTIONS)
		
		hidden_vals = tf.layers.dense(
			x,
			units=128,
			activation=tf.nn.relu)
		value = tf.layers.dense(
			hidden_vals,
			units=1)
		
		action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
		
		self.inputs = inputs
		self.logits = logits
		self.action = action
		self.value = value
		return logits, action, value
		
	def action_value(self, session, obs):
		logits, value = session.run([self.logits, self.value], feed_dict={self.inputs: obs})
		action = self.dist.predict(logits)
		return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
		
class A2CAgent:
	def __init__(self):
		self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
		self.model = Model()
		self.env = gym.make('CartPole-v0')
		self.S = tf.placeholder(tf.float32, [None, NUM_INPUTS])
		self.P_L, self.P_A, self.P_V = self.model(self.S)
		value_cost = self.value_cost()
		logits_cost = self.logits_cost()
		self.cost = value_cost + logits_cost
		self.train_op = tf.train.RMSPropOptimizer(7e-4).minimize(self.cost)
		self.session
		
	def test(self, env, render=True):
		obs = env.reset()
		done = False
		ep_reward = 0
		while not done:
			action, _ = self.model.action_value(obs[None, :])
			obs, reward, done, _ = env.step(action)
			ep_reward += reward
		return ep_reward
		
	def value_cost(self, returns, value):
		return self.params['value'] * tf.reduece_mean(tf.square(returns - value))
		
	def logits_cost(self, acts_and_advs, logits):
		actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
		weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
		actions = tf.cast(actions, tf.int32)
		policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
		entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
		return policy_loss - self.params['entropy'] * entropy_loss
		
	def train(self, env, batch_size=32, updates=1000):
		actions = np.empty((batch_size), dtype=np.int32)
		rewards, dones, values = np.empty((3, batch_size))
		observations = np.empty((batch_size,) + env.observation_space.shape)
		eps_rews = [0.0]
		next_obs = env.reset()
		for update in range(updates):
			for step in range(batch_size):
				observations[step] = next_obs.copy()
				actions[step], values[step] = self.model.action_value(self.session, next_obs[None,:])
				next_obs, rewards[step], dones[step], _ = env.step(actions[step])
				eps_rews[-1]+=rewards[step]
				if dones[step]:
					print('Update {:03d} Reward {:06f}'.format(update, eps_rews[-1]))
					eps_rews.append(0.0)
					next_obs = env.reset()
			_,next_value = self.model.action_value(self.session, next_obs[None, :])
			returns, advs = self._returns_advantages(rewards, dones, values, next_value)
			acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
			losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
		return eps_rews
	
	def _returns_advantages(self, rewards, dones, values, next_value):
		returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
		for t in reversed(range(rewards.shape[0])):
			returns[t] = rewards[t] + self.params['gamma'] * returns[t+1]*(1-dones[t])
		returns = returns[:-1]
		advantages = returns - values
		return returns, advantages

agent = A2CAgent()		

rewards_history = agent.train(env, batch_size=10)
print("Finished training, testing...")
#print("%d out of 200" % agent.test(env)) # 200 out of 200
