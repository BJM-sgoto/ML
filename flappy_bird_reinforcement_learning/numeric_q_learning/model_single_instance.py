import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random
from env import QuickEnvironment

NUM_EPISODES = 10000
NUM_SAMPLES = 50
NUM_ACTIONS = 2

#hyper parameter
GAMMA = 0.99
LAMDA = 0.0001
MAX_EPSILON = 1.0000
MIN_EPSILON = 0.0001

tf.disable_v2_behavior()

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
	
	MAX_GAME_DURATION = 100 # sec

	def __init__(self, sprite_folder='./images/', training=True):
		self.reuse_encoder = False
		self.n_actions = 2
		self.n_inputs = 3
		self.X = tf.placeholder(tf.float32, [None, self.n_inputs])
		self.Y = tf.placeholder(tf.float32, [None, self.n_actions])
		self.PY = self.encode(self.X, training=training)
		self.cost = self.compute_cost(self.PY, self.Y)
		self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		self.train_op = tf.group([self.train_op, update_op])
		self.env = QuickEnvironment(sprite_folder)
		#self.env = gym.make('CartPole-v1')
		self.memory = Memory()
		self.session = tf.Session()
		self.saver =  tf.train.Saver()
		self.epsilon = MAX_EPSILON
	
	def choose_action(self, state):
		if np.random.uniform()<self.epsilon:
			return np.random.randint(0, self.n_actions)
		else:
			return np.argmax(self.session.run(self.PY, feed_dict={self.X: np.reshape(state, [-1, self.n_inputs])})[0])
	
	def train_batch(self):
		samples = self.memory.sample(NUM_SAMPLES)
		states = np.float32([x[1] for x in samples])
		curr_rewards = self.session.run(self.PY, feed_dict={self.X: states})
		next_states = np.float32([np.zeros([self.n_inputs], dtype=np.float32) if x[4] is None else x[4] for x in samples])
		next_rewards = self.session.run(self.PY, feed_dict={self.X: next_states})
		for i, sample in enumerate(samples):
			if sample[4] is None:
				curr_rewards[i, sample[2]] = sample[3]
			else:
				curr_rewards[i, sample[2]] = sample[3] + GAMMA * np.amax(next_rewards[i])
		cost_val, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X: states, self.Y: curr_rewards})
		return cost_val
	
	def encode(self, input_holder, training=True):
		output_holder = input_holder
		with tf.variable_scope('encoder', reuse = self.reuse_encoder):
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.leaky_relu)
			output_holder = tf.layers.dense(
				output_holder,
				units=64,
				activation=tf.nn.leaky_relu)
			output_holder = tf.layers.dense(
				output_holder,
				units=NUM_ACTIONS)
			self.reuse_encoder = True
			return output_holder
			
	def compute_cost(self, predicted_overall_reward, overall_rewards):
		cost = tf.reduce_mean(tf.square(predicted_overall_reward - overall_rewards))
		return cost		

	def train(self, model_path='./model/model', resume=False):
		step = 0
		if resume:
			self.saver.restore(self.session, model_path)
		else:
			self.session.run(tf.global_variables_initializer())
		for episode_id in range(NUM_EPISODES):
			done = False
			state = self.env.reset()
			average_loss = 0
			duration = 0
			while not done:
				duration += 1
				step += 1
				self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-LAMDA * step)
				action = self.choose_action(state)
				next_state, reward, done = self.env.update(action)
				if done:
					next_state = None
				self.memory.add_sample([episode_id, state, action, reward, next_state, None])
				state = next_state
				loss_val = self.train_batch()
				average_loss += loss_val
			average_loss = average_loss/duration
			print('Episode {:04d}, Duration {:04d}, Epsilon {:05f}, Average Loss {:05f}'.format(episode_id, duration, self.epsilon, average_loss))
			if (episode_id + 1)%100==0:
				self.saver.save(self.session, model_path)
		self.session.close()
	
model = Model(sprite_folder='./images/', training=True)
model.train(model_path='./model_single_instance/model', resume=False)