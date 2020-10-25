import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import random
import datetime
from env import QuickEnvironment, WIN_WIDTH, WIN_HEIGHT

NUM_EPISODES = 2000
NUM_SAMPLES = 50
NUM_FRAMES = 4
NUM_ACTIONS = 2

FRAME_WIDTH = WIN_WIDTH
FRAME_HEIGHT = WIN_HEIGHT

RESIZED_FRAME_WIDTH = 160
RESIZED_FRAME_HEIGHT = 128

#hyper parameter
GAMMA = 0.99
LAMDA = 0.0001
MAX_EPSILON = 0.2500
MIN_EPSILON = 0.0001

tf.disable_v2_behavior()

base_folder = './'
#base_folder = '/content/gdrive/My Drive/machine_learning_data/reinforcement_learning/'

class Memory:
	def __init__(self, max_memory=500): # consume 1GB
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
		self.env = QuickEnvironment(sprite_folder)
		self.memory = Memory()
		self.session = tf.Session()
		self.saver =  tf.train.Saver()
		self.epsilon = MAX_EPSILON
		if not training:
			self.epsilon = MIN_EPSILON
	
	def choose_action(self, state):
		if np.random.uniform()<self.epsilon:
			return np.random.randint(0, self.n_actions)
		else:
			return np.argmax(self.session.run(self.PY, feed_dict={self.X: np.expand_dims(state, axis=0)})[0])
	
	def train_batch(self):
		if self.memory.get_size()<=1:
			return 0
		start = datetime.datetime.now()
		curr_states, actions, rewards, dones, next_states = self.memory.sample(NUM_SAMPLES)
		end = datetime.datetime.now()
		print('Delta time0', end - start)
		start = datetime.datetime.now()
		curr_rewards = self.session.run(self.PY, feed_dict={self.X: curr_states})
		end = datetime.datetime.now()
		print('Delta time1', end - start)
		start = datetime.datetime.now()
		next_rewards = self.session.run(self.PY, feed_dict={self.X: next_states})
		end = datetime.datetime.now()
		print('Delta time2', end - start)
		for i, done in enumerate(dones):
			if done:
				curr_rewards[i, actions[i]] = rewards[i]
			else:
				curr_rewards[i, actions[i]] = rewards[i] + GAMMA * np.amax(next_rewards[i])
		start = datetime.datetime.now()
		cost_val, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X: curr_states, self.Y: curr_rewards})
		end = datetime.datetime.now()
		print('Delta time3', end - start)
		return cost_val
	
	def encode(self, input_holder, training=True):
		output_holder = input_holder/255
		output_holder = tf.image.resize(
			output_holder,
			size=(RESIZED_FRAME_HEIGHT,RESIZED_FRAME_HEIGHT))
		with tf.variable_scope('encoder', reuse = self.reuse_encoder):
			layer_depths = [32, 32, 64, 64, 128, 8]
			for layer_depth in layer_depths:
				output_holder = tf.layers.conv2d(
					output_holder,
					kernel_size=(3,3),
					filters=layer_depth,
					padding='same',
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
		cost = tf.reduce_mean(tf.square(predicted_overall_reward - overall_rewards))
		return cost		

	def train(self, model_path='./model/model', resume=False):
		step = 0
		stacked_state = []
		if resume:
			self.saver.restore(self.session, model_path)
		else:
			self.session.run(tf.global_variables_initializer())
		for episode_id in range(NUM_EPISODES):
			done = False
			state = self.env.reset()
			stacked_state = [state]*NUM_FRAMES
			average_loss = 0
			duration = 0
			while not done:
				print('--------------------')
				start = datetime.datetime.now()
				duration += 1
				step += 1
				self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-LAMDA * step)
				action = self.choose_action(np.concatenate(stacked_state, axis=2))
				next_state, reward, done = self.env.update(action)
				self.memory.add_sample([episode_id, state, action, reward, done])
				state = next_state
				stacked_state.pop(0)
				stacked_state.append(state)
				loss_val = self.train_batch()
				average_loss += loss_val
				print('Episode {:04d}, Epsilon {:05f}, Loss {:05f}'.format(episode_id, self.epsilon, loss_val))
				end = datetime.datetime.now()
				print('Delta time4', end-start)
			average_loss = average_loss/duration
			print('* Episode {:04d}, Duration {:04d}, Epsilon {:05f}, Average Loss {:05f}'.format(episode_id, duration, self.epsilon, average_loss))
			if (episode_id+1)%5==0:
				print('--------- Save ---------')
				self.saver.save(self.session, model_path)
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
model.train(model_path=base_folder+'model_single_instance/model', resume=False)

'''
model = Model(sprite_folder = base_folder + 'cimages/', training = False)
model.test(model_path = base_folder + 'model/model')
'''