import numpy as np
import pandas as pd

SEED = None
SEED_TO_SPLIT_DATA = 5412

np.random.seed(SEED)

NUM_NUMERICAL_ATTR = 872
NUM_SAMPLED_NUMERICAL_ATTR = 218 # 25% of NUM_CATEGORICAL_ATTR
NUM_CATEGORICAL_ATTR = 3
NUM_SCORED_OUTPUT = 206
NUM_NONSCORED_OUTPUT = 402
NUM_SPLITS = 50 - 1
SMALL_NUMBER = 1e-4



class TreeMaker:
	
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
	def find_splitpoint(subdataset):
		indices, categorical_features, numerical_features, scored_targets, nonscored_targets = subdataset
		targets = scored_targets
		min_var = NUM_SCORED_OUTPUT + NUM_NONSCORED_OUTPUT
		split_position = -10
		split_attr = -10
		split_points = 2/(NUM_SPLITS+1)*np.arange(NUM_SPLITS)-1
		x = np.concatenate([categorical_features, numerical_features], axis=1)
		y = np.expand_dims(targets, axis=1) # B X 1 X 600
		
		for var_count in range(NUM_CATEGORICAL_ATTR + NUM_NUMERICAL_ATTR):
			attr = var_count
			upper_points = np.greater_equal(x[:, attr:attr+1], split_points)
			upper_points = np.expand_dims(upper_points, axis=2) # B X 50 X 1
			upper_points = np.float32(upper_points) # B X 50 X 1
			lower_points = 1 - upper_points # B X 50 X 1
			
			# number of points
			n_upper_points = np.sum(upper_points, axis=0) # 50 X 1
			n_lower_points = np.sum(lower_points, axis=0) # 50 X 1
			
			# sum
			upper_y = y * upper_points # B X 50 X 600
			lower_y = y - upper_y # B X 500 X 600
			upper_sum = np.sum(upper_y, axis=0) # 50 X 600
			lower_sum = np.sum(lower_y, axis=0) # 50 X 600
			
			# mean
			upper_mean = upper_sum/(n_upper_points+SMALL_NUMBER) # 50 X 600
			lower_mean = lower_sum/(n_lower_points+SMALL_NUMBER) # 50 X 600
			
			# variance
			upper_variance = np.sum(np.square(y-upper_mean)*upper_points, axis=0)/(n_upper_points+SMALL_NUMBER) # 50 X 600
			lower_variance = np.sum(np.square(y-lower_mean)*lower_points, axis=0)/(n_lower_points+SMALL_NUMBER) # 50 X 600
			variance = np.sum(upper_variance+lower_variance, axis=1) # 50
			current_split_position = np.argmin(variance) # ()
			current_min_var = variance[current_split_position]
			current_mean_var = np.mean(variance)
			if current_min_var<min_var:
				min_var = current_min_var
				split_position = current_split_position
				split_attr = attr

			print('count {:03d}, min {:06f}, cur_min {:06f}, cur_mean {:06f}'.format(var_count, min_var, current_min_var, current_mean_var))
			
		return split_position, split_attr, min_var


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
		n_train_data = len(train_dataset[0])
		subdataset = TreeMaker.make_subdataset(train_dataset, random_seed=1, subdataset_size=2000)
		split_point, split_attr, min_var = TreeMaker.find_splitpoint(subdataset)
		print(split_point, split_attr, min_var)

model = GB()
model.train(
	train_feature_file='./train_features.csv', 
	scored_target_file='./train_targets_scored.csv', 
	nonscored_target_file='./train_targets_nonscored.csv')