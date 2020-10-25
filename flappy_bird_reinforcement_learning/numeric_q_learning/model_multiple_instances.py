import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random
from env import QuickEnvironment

NUM_EPISODES = 10000
NUM_SAMPLES = 50
NUM_GAMES = 10
NUM_ACTIONS = 2

#hyper parameter
GAMMA = 0.99
LAMDA = 0.0001
MAX_EPSILON = 1.0000
MIN_EPSILON = 0.0001

tf.disable_v2_behavior()

class Memory:
	def __init__(self, max_memory=1000):
		self.max_memory = max_memory
		self.memory = []

	# sample format :  [state, action, reward, next_state]
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
		self.saver =  tf.train.Saver()
		self.epsilon = MAX_EPSILON
		self.session = tf.Session()
		
		self.env = []
		self.memory = []
		for i in range(NUM_GAMES):
			self.env.append(QuickEnvironment(sprite_folder))
			self.memory.append(Memory())
	
	def choose_action(self, state):
		random_rate = np.random.uniform(size=[NUM_GAMES])
		random_action = np.random.randint(low=0, high=self.n_actions, size=[NUM_GAMES])
		action = np.argmax(self.session.run(self.PY, feed_dict={self.X: state}), axis=1)
		for i in range(NUM_GAMES):
			if random_rate[i]<self.epsilon:
				action[i] = random_action[i]
		return action
	
	def train_batch(self, game_id):
		samples = self.memory[game_id].sample(NUM_SAMPLES)
		states = np.float32([x[0] for x in samples])
		curr_rewards = self.session.run(self.PY, feed_dict={self.X: states})
		next_states = np.float32([np.zeros([self.n_inputs], dtype=np.float32) if x[3] is None else x[3] for x in samples])
		next_rewards = self.session.run(self.PY, feed_dict={self.X: next_states})
		for i, sample in enumerate(samples):
			if sample[3] is None:
				curr_rewards[i, sample[1]] = sample[2]
			else:
				curr_rewards[i, sample[1]] = sample[2] + GAMMA * np.amax(next_rewards[i])
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
		
		count_game = 0
		state = [None for i in range(NUM_GAMES)]# np.zeros([NUM_GAMES, self.n_inputs], dtype=np.float32)
		reward = [0 for i in range(NUM_GAMES)]
		done = [True for i in range(NUM_GAMES)] # set True to reset game
		next_state = [None for i in range(NUM_GAMES)] # np.zeros([NUM_GAMES, self.n_inputs], dtype=np.float32)
		duration = [0 for i in range(NUM_GAMES)]
		saved = False
		while True:
			
			for game_id in range(NUM_GAMES):
				if done[game_id]:
					state[game_id] = self.env[game_id].reset()
					done[game_id] = False
					duration[game_id] = 0
					
			action = self.choose_action(state)
			
			for game_id in range(NUM_GAMES):
				step += 1
				self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-LAMDA * step)
				next_state[game_id], reward[game_id], done[game_id] = self.env[game_id].update(action[game_id])
				self.memory[game_id].add_sample([state[game_id], action[game_id], reward[game_id], next_state[game_id]])
				duration[game_id]+=1
				if done[game_id]:
					next_state[game_id] = None
					count_game += 1
					saved = False
					print('Game {:02d} duration {:04d} epsilon {:06f} count {:05d}'.format(game_id, duration[game_id], self.epsilon, count_game))
				state[game_id] = next_state[game_id]
				self.train_batch(game_id)
			if count_game >= NUM_GAMES*NUM_EPISODES:
				break
			if (count_game+1)%100==0 and not saved:
				print('------ Save ------')
				self.saver.save(self.session, model_path)
				saved = True
		self.session.close()
				
model = Model(sprite_folder='./images/', training=True)
model.train(model_path='./model_multiple_instances/model', resume=False)