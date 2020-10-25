import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd
import datetime

SEED = 1
SEED_TO_SPLIT_DATA = 5412
tf.disable_v2_behavior()
tf.reset_default_graph()

tf.set_random_seed(SEED)
np.random.seed(SEED)

NUM_NUMERICAL_ATTR = 872
NUM_CATEGORICAL_ATTR = 3
NUM_SCORED_OUTPUT = 206
NUM_NONSCORED_OUTPUT = 402
NUM_SPLITS = 50 - 1
SMALL_NUMBER = 1e-4


class TreeMaker:
	SESSION = None
	XN = None
	XC = None
	X = None
	Y1 = None
	Y2 = None
	Y = None
	SPLIT_POSITION = None
	SPLIT_ATTRIBUTE = None

	ENTROPY = None

	@staticmethod
	def initialize():
		TreeMaker.SESSION = tf.Session()
		TreeMaker.XC = tf.placeholder(tf.float32, shape=[None, NUM_CATEGORICAL_ATTR])
		TreeMaker.XN = tf.placeholder(tf.float32, shape=[None, NUM_NUMERICAL_ATTR])
		TreeMaker.X = tf.concat([TreeMaker.XC, TreeMaker.XN], axis=1)
		TreeMaker.Y1 = tf.placeholder(tf.float32, shape=[None, NUM_SCORED_OUTPUT])
		TreeMaker.Y2 = tf.placeholder(tf.float32, shape=[None, NUM_NONSCORED_OUTPUT])
		#TreeMaker.Y = tf.concat([TreeMaker.Y1, TreeMaker.Y2], axis=1)
		TreeMaker.Y = TreeMaker.Y1
		TreeMaker.SPLIT_POSITION, TreeMaker.SPLIT_ATTRIBUTE, TreeMaker.ENTROPY = TreeMaker._make_model()
		

	@staticmethod
	def close():
		TreeMaker.SESSION.close()
		TreeMaker.SESSION = None
		TreeMaker.XC = None
		TreeMaker.XN = None
		TreeMaker.X = None
		TreeMaker.Y1 = None
		TreeMaker.Y2 = None
		TreeMaker.Y = None
		TreeMaker.SPLIT_POSITION = None
		TreeMaker.SPLIT_ATTRIBUTE = None
		TreeMaker.ENTROPY = None

	@staticmethod
	def make_subdataset(dataset, random_seed, subdataset_size):
		np.random.seed(random_seed)
		indices, categorical_features, numerical_features, scored_targets, nonscored_targets = dataset
		n_data = len(indices)
		ids = np.arange(n_data)
		np.random.shuffle(ids)
		sub_indices = []
		sub_categorical_features = []
		sub_numerical_featuers = []
		sub_scored_targets = []
		sub_nonscored_targets = []
		subdataset_size = min(n_data, subdataset_size)
		for id in ids[:subdataset_size]:
			sub_indices.append(indices[id])
			sub_categorical_features.append(categorical_features[id])
			sub_numerical_featuers.append(numerical_features[id])
			sub_scored_targets.append(scored_targets[id])
			sub_nonscored_targets.append(nonscored_targets[id])
		sub_indices = np.array(indices)
		sub_categorical_features = np.int32(sub_categorical_features)
		sub_numerical_featuers = np.float32(sub_numerical_featuers)
		sub_scored_targets = np.float32(sub_scored_targets)
		sub_nonscored_targets = np.float32(sub_nonscored_targets)
		return sub_indices, sub_categorical_features, sub_numerical_featuers, sub_scored_targets, sub_nonscored_targets

	@staticmethod
	def _make_model():
		count = tf.constant(0, dtype=tf.int32)
		min_variance = tf.constant(NUM_NUMERICAL_ATTR, dtype=tf.float32)
		split_position = tf.constant(-10, dtype=tf.int32)
		split_attr = tf.constant(-10, dtype=tf.int32)
		split_points = 2/(NUM_SPLITS+1)*np.arange(NUM_SPLITS)-1 # 50
		split_points = np.expand_dims(split_points, axis=0) # 1 X 50
		y = tf.expand_dims(TreeMaker.Y, axis=1) # B X 1 X 600
		all_sum = tf.reduce_sum(y, axis=0) # 1 X 600
		n_points = tf.cast(tf.shape(y)[0], dtype=tf.float32) # 1
		def cond(var_count, var_min_variance, var_split_pos, var_split_attr, const_y, const_all_sum, const_n_points):
			return tf.less(var_count, NUM_CATEGORICAL_ATTR + NUM_NUMERICAL_ATTR)
			#return tf.less(var_count, 400)

		def body(var_count, var_min_variance, var_split_pos, var_split_attr, const_y, const_all_sum, const_n_points):
			
			upper_points = tf.greater_equal(TreeMaker.X[:, var_count: var_count+1], split_points) # B X 50
			upper_points = tf.expand_dims(upper_points, axis=2) # B X 50 X 1
			upper_points = tf.cast(upper_points, dtype=tf.float32) # B X 50 X 1
			lower_points = 1 - upper_points # B X 100 X 1

			# number of points
			n_upper_points = tf.reduce_sum(upper_points, axis=0) # 50 X 1
			n_lower_points = const_n_points - n_upper_points # 50 X 1

			# sum
			upper_sum = tf.reduce_sum(const_y*upper_points, axis=0) # 50 X 600
			lower_sum = const_all_sum - upper_sum # 50 X 600

			# prob
			upper_prob = upper_sum/(n_upper_points+SMALL_NUMBER) # 50 X 600
			reverse_upper_prob = 1-upper_prob
			lower_prob = lower_sum/(n_lower_points+SMALL_NUMBER) # 50 X 600
			reverse_lower_prob = 1-lower_prob

			# entropy
			upper_entropy = -(upper_prob*tf.log(upper_prob+SMALL_NUMBER) + reverse_upper_prob*tf.log(reverse_upper_prob+SMALL_NUMBER))#50X600
			lower_entropy = -(lower_prob*tf.log(lower_prob+SMALL_NUMBER) + reverse_lower_prob*tf.log(reverse_lower_prob+SMALL_NUMBER))#50X600
			entropy = tf.reduce_sum(upper_entropy+lower_entropy, axis=1)
			entropy = tf.reduce_min(entropy)
			var_min_variance = entropy
			var_count = var_count + 1
			return var_count, var_min_variance, var_split_pos, var_split_attr, const_y, const_all_sum, const_n_points

		_, entropy_val, split_position, split_attr, _, _, _ = tf.while_loop(cond, body, [count, min_entropy, split_position, split_attr, y, all_sum, n_points],back_prop=False)
		return entropy_val, split_position, split_attr

	@staticmethod
	def find_splitpoint(subdataset):
		_, categorical_features, numerical_features, scored_targets, nonscored_targets = subdataset
		start = datetime.datetime.now()
		entropy_val, _, _ = TreeMaker.SESSION.run([TreeMaker.SPLIT_POSITION, TreeMaker.SPLIT_ATTRIBUTE, TreeMaker.ENTROPY], feed_dict={TreeMaker.XC: categorical_features, TreeMaker.XN: numerical_features, TreeMaker.Y1: scored_targets, TreeMaker.Y2: nonscored_targets})
		end = datetime.datetime.now()
		print(end - start, entropy_val)		


class GB:
	def __init__(self):
		self.categorical_attrs = {
			'cp_type': ['trt_cp', 'ctl_vehicle'], 
			'cp_time': [24, 48, 72], 
			'cp_dose': ['D1', 'D2']}

	def make_dataset(self, input_feature_file, scored_target_file=None, nonscored_target_file=None):
		input_features = pd.read_csv(input_feature_file)
		n_data = len(input_features)
		# seperate index
		indices = input_features['sig_id'].values
		input_features = input_features.drop('sig_id', axis=1)
		feature_columns = list(input_features.columns.values)
		n_features = len(feature_columns)
		# make categorical attributes
		categorical_features = np.zeros([n_data, 3], dtype=np.float32)
		for i, categorical_attr in enumerate(self.categorical_attrs.keys()):
			temp_datas = input_features[categorical_attr].values
			data_pos = {}
			for j, attr_val in enumerate(self.categorical_attrs[categorical_attr]):
				data_pos[attr_val] = j
			for j in range(n_data):
				categorical_features[j,i] = data_pos[temp_datas[j]]
		categorical_features[:,0] = categorical_features[:,0] * 2 - 1
		categorical_features[:,1] = categorical_features[:,1] - 1
		categorical_features[:,2] = categorical_features[:,2] * 2 - 1

		# make numerical attributes
		feature_columns.remove('cp_type')
		feature_columns.remove('cp_time')
		feature_columns.remove('cp_dose')
		n_features = len(feature_columns)
		numerical_features = np.zeros([n_data, n_features], dtype=np.float32)
		for i, numerical_attr in enumerate(feature_columns):
			numerical_features[:, i] = input_features[numerical_attr].values

		### normalize ## TODO: fix????
		numerical_features = numerical_features/10 

		# make scored targets
		scored_targets = None
		if not scored_target_file is None:
			raw_scored_targets = pd.read_csv(scored_target_file)
			raw_scored_targets = raw_scored_targets.drop('sig_id', axis=1)
			feature_columns = list(raw_scored_targets.columns.values)
			n_features = len(feature_columns)
			scored_targets = np.zeros([n_data, n_features], dtype=np.float32)
			for i, attr in enumerate(feature_columns):
				scored_targets[:, i] = raw_scored_targets[attr].values
		
		nonscored_targets = None
		if not nonscored_target_file is None:
			raw_nonscored_targets = pd.read_csv(nonscored_target_file)
			raw_nonscored_targets = raw_nonscored_targets.drop('sig_id', axis=1)
			feature_columns = list(raw_nonscored_targets.columns.values)
			n_features = len(feature_columns)
			nonscored_targets = np.zeros([n_data, n_features], dtype=np.float32)
			for i, attr in enumerate(feature_columns):
				nonscored_targets[:, i] = raw_nonscored_targets[attr].values

		return indices, categorical_features, numerical_features, scored_targets, nonscored_targets

	def shuffle_dataset(self, dataset):
		indices, categorical_features, numerical_features, scored_targets, nonscored_targets = dataset
		n_data = len(indices)
		ids = np.arange(n_data)
		np.random.shuffle(ids)
		new_indices = []
		new_categorical_features = []
		new_numerical_features = []
		new_scored_targets = []
		new_nonscored_targets = []
		for i in ids:
			new_indices.append(indices[i])
			new_categorical_features.append(categorical_features[i])
			new_numerical_features.append(numerical_features[i])
			new_scored_targets.append(scored_targets[i])
			new_nonscored_targets.append(nonscored_targets[i])
		new_indices = np.array(new_indices)
		new_categorical_features = np.float32(new_categorical_features)
		new_numerical_features = np.float32(new_numerical_features)
		new_scored_targets = np.int32(new_scored_targets)
		new_nonscored_targets = np.int32(new_nonscored_targets)
		return new_indices, new_categorical_features, new_numerical_features, new_scored_targets, new_nonscored_targets

	def split_dataset(self, dataset, test_rate=0.2):
		np.random.seed(SEED_TO_SPLIT_DATA)
		indices, categorical_features, numerical_features, scored_targets, nonscored_targets = self.shuffle_dataset(dataset)
		n_data = len(indices)
		train_size = int(n_data * (1 - test_rate))

		train_indices = indices[:train_size]
		train_categorical_features = categorical_features[:train_size]
		train_numerical_features = numerical_features[:train_size]
		train_scored_targets = scored_targets[:train_size]
		train_nonscored_targets = nonscored_targets[:train_size]

		test_indices = indices[train_size:]
		test_categorical_features = categorical_features[train_size:]
		test_numerical_features = numerical_features[train_size:]
		test_scored_targets = scored_targets[train_size:]
		test_nonscored_targets = nonscored_targets[train_size:]

		np.random.seed(SEED)
		return (train_indices, train_categorical_features, train_numerical_features, train_scored_targets, train_nonscored_targets), (test_indices, test_categorical_features, test_numerical_features, test_scored_targets, test_nonscored_targets)

	def train(self, train_feature_file, scored_target_file, nonscored_target_file):
		dataset = self.make_dataset(train_feature_file, scored_target_file, nonscored_target_file)
		train_dataset, valid_dataet = self.split_dataset(dataset, test_rate=0.1)
		n_train_data=len(train_dataset[0])
		TreeMaker.initialize()
		subdataset = TreeMaker.make_subdataset(train_dataset, random_seed=1, subdataset_size=2000)
		for i in range(10):
			TreeMaker.find_splitpoint(subdataset)
		TreeMaker.close()



model = GB()
model.train(
	train_feature_file='./train_features.csv', 
	scored_target_file='./train_targets_scored.csv', 
	nonscored_target_file='./train_targets_nonscored.csv')