import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random
from env import QuickEnvironment, WIN_WIDTH, WIN_HEIGHT

NUM_EPISODES = 2000
NUM_SAMPLES = 10 # => batchsize = NUM_SAMPLES when training
NUM_FRAMES = 4
NUM_ACTIONS = 2
NUM_GAMES = NUM_SAMPLES # => run NUM_GAMES games, and pick NUM_GAMES actions at the same time

FRAME_WIDTH = WIN_WIDTH
FRAME_HEIGHT = WIN_HEIGHT

#hyper parameter
GAMMA = 0.99
LAMDA = 0.001
MAX_EPSILON = 0.0002
MIN_EPSILON = 0.0001

tf.disable_v2_behavior()

base_folder = './'
#base_folder = '/content/gdrive/My Drive/machine_learning_data/reinforcement_learning/'

class Memory:
	def __init__(self, max_memory=200): # consume 1GB
		self.max_memory = max_memory
		self.memory = []

	# sample format :  [episode_id, state, action, reward, done]
	def add_sample(self, sample):
		self.memory.append(sample)
		if len(self.memory)>self.max_memory:
			self.memory.pop(0)
	
	def get_size(self):
		return len(self.memory)
	
	def sample(self, n_samples):
		mem_size = len(self.memory)
		choices = random.sample(range(mem_size-1), min(n_samples, mem_size-1)) # do not pick the last frame, if so there won't be the next frame
		curr_states = []
		actions = [self.memory[i][2] for i in choices]
		rewards = [self.memory[i][3] for i in choices]
		next_states = []
		dones = [self.memory[i][4] for i in choices]
		for pos in choices:
			# get current states
			curr_state = []
			sample = self.memory[pos]
			for i in range(NUM_FRAMES):
				temp_pos = pos - i
				if temp_pos>=0 and self.memory[temp_pos][0] == sample[0]:
					curr_state.insert(0, self.memory[temp_pos][1])
				else:
					break
			curr_state = [curr_state[0]]*(NUM_FRAMES - len(curr_state)) + curr_state
			
			# get next states
			next_state = curr_state[1:]
			if sample[4]==True or sample[0]!=self.memory[pos+1][0]: # game was done, there is no next state
				next_state.append(np.zeros([FRAME_HEIGHT, FRAME_WIDTH, 3], dtype=np.float32))
			else:
				next_state.append(self.memory[pos+1][1])
			
			curr_state = np.concatenate(curr_state, axis=2)
			next_state = np.concatenate(next_state, axis=2)
			curr_states.append(curr_state)
			next_states.append(next_state)
		curr_states = np.float32(curr_states)
		next_states = np.float32(next_states)
		return curr_states, actions, rewards, dones, next_states
	
class Model:
	
	MAX_GAME_DURATION = 100 # sec

	def __init__(self, sprite_folder='./images/', training=True):
		self.reuse_encoder = False
		self.n_actions = 2
		self.n_inputs = 3
		self.X = tf.placeholder(tf.float32, [None, FRAME_HEIGHT, FRAME_WIDTH, NUM_FRAMES*3])
		self.Y = tf.placeholder(tf.float32, [None, self.n_actions])
		self.PY = self.encode(self.X, training=training)
		self.cost = self.compute_cost(self.PY, self.Y)
		self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		self.train_op = tf.group([self.train_op, update_op])
		self.env = []
		self.memory = []
		for i in range(NUM_GAMES):
			self.env.append(QuickEnvironment(sprite_folder))
			self.memory.append(Memory())
		self.session = tf.Session()
		self.saver =  tf.train.Saver()
		self.epsilon = MAX_EPSILON
		if not training:
			self.epsilon = MIN_EPSILON
	
	def choose_action(self, state):
		if np.random.uniform()<self.epsilon:
			return np.random.randint(low=0, high=self.n_actions, size=[len(state)])
		else:
			return np.argmax(self.session.run(self.PY, feed_dict={self.X: state}), axis=1)
	
	def train_batch(self):
		curr_states, actions, rewards, dones, next_states = [],[],[],[],[]
		for i in range(NUM_GAMES):
			if self.memory[i].get_size()>1:
				curr_state, action, reward, done, next_state = self.memory[i].sample(1)
				curr_states.append(curr_state)
				actions.append(action)
				rewards.append(reward)
				dones.append(done)
				next_states.append(next_state)
		if len(curr_states)==0:
			return np.zeros([NUM_GAMES], dtype=np.float32)
		curr_states = np.concatenate(curr_states, axis=0)
		next_states = np.concatenate(next_states, axis=0)
		
		curr_rewards = self.session.run(self.PY, feed_dict={self.X: curr_states})
		next_rewards = self.session.run(self.PY, feed_dict={self.X: next_states})
		for i, done in enumerate(dones):
			if done:
				curr_rewards[i, actions[i]] = rewards[i]
			else:
				curr_rewards[i, actions[i]] = rewards[i] + GAMMA * np.amax(next_rewards[i])
		cost_val, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X: curr_states, self.Y: curr_rewards})
		return cost_val
	
	def encode(self, input_holder, training=True):
		output_holder = input_holder
		with tf.variable_scope('encoder', reuse = self.reuse_encoder):
			output_holder = tf.layers.conv2d(
				output_holder,
				filters=32,
				kernel_size=(3,3),
				strides=(2,2),
				padding='valid',
				activation=tf.nn.leaky_relu)
			output_holder = tf.layers.batch_normalization(
				output_holder,
				training=training)
			layer_depths = [32, 64, 64, 128, 8]
			for layer_depth in layer_depths:
				output_holder = tf.layers.conv2d(
					output_holder,
					kernel_size=(3,3),
					filters=layer_depth,
					padding='valid',
					activation=tf.nn.leaky_relu)
				output_holder = tf.layers.max_pooling2d(
					output_holder,
					strides=(2,2),
					pool_size=(2,2))
				output_holder = tf.layers.batch_normalization(
					output_holder,
					training=training)
			output_holder = tf.layers.flatten(output_holder)
			output_holder = tf.layers.dense(
				output_holder,
				units=32,
				activation = tf.nn.leaky_relu)
			output_holder = tf.layers.dense(
				output_holder,
				units=NUM_ACTIONS)
			self.reuse_encoder = True
			return output_holder
			
	def compute_cost(self, predicted_overall_reward, overall_rewards):
		cost = tf.reduce_mean(tf.square(predicted_overall_reward - overall_rewards),axis=1)
		return cost		

	def train(self, model_path='./model/model', resume=False):
		step = 0
		stacked_state = []
		if resume:
			self.saver.restore(self.session, model_path)
		else:
			self.session.run(tf.global_variables_initializer())
		count_game = np.zeros([NUM_GAMES], dtype=np.int32)
		state = [None for i in range(NUM_GAMES)]
		action = [0 for i in range(NUM_GAMES)]
		reward = [0 for i in range(NUM_GAMES)]
		done = [True for i in range(NUM_GAMES)]
		next_state = [None for i in range(NUM_GAMES)]
		stacked_state = [[] for i in range(NUM_GAMES)]
		duration = [0 for i in range(NUM_GAMES)]
		loss = np.zeros([NUM_GAMES], dtype=np.float32)
		while True:
			step += 1
			self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-LAMDA * step)
			
			for game_id in range(NUM_GAMES):
				if done[game_id]:
					state[game_id] = self.env[game_id].reset()
					stacked_state[game_id] = [state[game_id]] * NUM_FRAMES
					duration[game_id] = 0
					done[game_id] = False
					count_game[game_id] += 1
					loss[game_id] = 0

			action = self.choose_action(np.float32([np.concatenate(x, axis=2) for x in stacked_state]))
			
			sum_count_game = np.sum(count_game)
			
			for game_id in range(NUM_GAMES):
				next_state[game_id], reward[game_id], done[game_id] = self.env[game_id].update(action[game_id])
				self.memory[game_id].add_sample([count_game[game_id], state[game_id], action[game_id], reward[game_id], done[game_id]])
				stacked_state[game_id].pop(0)
				stacked_state[game_id].append(next_state[game_id])
				state[game_id] = next_state[game_id]
				duration[game_id] += 1
				if done[game_id]:
					print('Game {:1d}, Duration {:04d}, Epsilon {:05f}, Count {:04d}, Loss {:04f}'.format(game_id, duration[game_id], self.epsilon, sum_count_game, loss[game_id]/duration[game_id]))
				
			loss += self.train_batch()
			
			if (sum_count_game+1)%100==0:
				self.saver.save(self.session, model_path)
			if sum_count_game> NUM_EPISODES:
				break	
		
		self.session.close()
		
		
	def test(self, model_path='./model/model'):
		self.saver.restore(self.session, model_path)
		state = self.env.reset()
		done = False
		count = 0
		while not done:
			action = self.choose_action(state)
			frame = self.env.screenshot()
			cv2.imwrite('./output/{:06d}.jpg'.format(count), frame)
			print(state, count)
			state, reward, done = self.env.update(action)
			count += 1


model = Model(sprite_folder=base_folder+'images/', training=True)
model.train(model_path=base_folder+'model/model', resume=True)

'''
model = Model(sprite_folder = base_folder + 'cimages/', training = False)
model.test(model_path = base_folder + 'model/model')
'''