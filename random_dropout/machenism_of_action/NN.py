import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd

SEED = None
tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(SEED)
np.random.seed(SEED)

SEED_TO_SPLIT_DATA = 1

HIDDEN_DROPRATE = 0.4
INPUT_DROPRATE = 0.5
NOISE_MAGNITUDE = 0.005

NUM_NUMERICAL_ATTR = 872
NUM_CATEGORICAL_ATTR = 3
NUM_SCORED_OUTPUT = 206
NUM_NONSCORED_OUTPUT = 402
NUM_DECISIONS = 20

NONSCORE_WEIGHT = 1.0

LEARNING_RATE = 1e-4
RESUME = True

class Model:
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
		categorical_features = np.zeros([n_data, 3])
		for i, categorical_attr in enumerate(self.categorical_attrs.keys()):
			temp_datas = input_features[categorical_attr].values
			data_pos = {}
			for j, attr_val in enumerate(self.categorical_attrs[categorical_attr]):
				data_pos[attr_val] = j
			for j in range(n_data):
				categorical_features[j,i] = data_pos[temp_datas[j]]
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
			scored_targets = np.zeros([n_data, n_features], dtype=np.int32)
			for i, attr in enumerate(feature_columns):
				scored_targets[:, i] = raw_scored_targets[attr].values
		
		nonscored_targets = None
		if not nonscored_target_file is None:
			raw_nonscored_targets = pd.read_csv(nonscored_target_file)
			raw_nonscored_targets = raw_nonscored_targets.drop('sig_id', axis=1)
			feature_columns = list(raw_nonscored_targets.columns.values)
			n_features = len(feature_columns)
			nonscored_targets = np.zeros([n_data, n_features], dtype=np.int32)
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

	def build_model(self, X_C, X_N, add_noise, drop_input, drop_hidden, return_nonscored_output=True):
		with tf.variable_scope('model'):
			kernel_regularizer = tf.keras.regularizers.L2(l2=0.01)
			bias_regularizer = tf.keras.regularizers.L2(l2=0.01)
			# 3 categorical attrs
			X_C1, X_C2, X_C3 = tf.split(X_C, num_or_size_splits=3, axis=1)
			X_C = X_C1*6 + X_C2*2 + X_C3
			K_N = tf.ones_like(X_N)

			# add noise
			add_noise = tf.cast(add_noise, dtype=tf.float32)
			noise = (tf.random.uniform(shape=tf.shape(X_N), minval=0, maxval=1) * 2 - 1) * NOISE_MAGNITUDE * add_noise
			X_N = X_N + noise
			
			# do not drop categorical attributes 
			# drop numerical attributes 
			ones = tf.ones_like(K_N)
			zeros = tf.zeros_like(K_N)
			K_N = tf.random.uniform(shape=tf.shape(X_N), minval=0, maxval=1, dtype=tf.float32)
			K_N = tf.where(tf.less(K_N, INPUT_DROPRATE), zeros, ones)
			drop_input = tf.cast(drop_input, dtype=tf.float32)
			K_N = K_N + ones * (1 - drop_input)
			K_N = tf.where(tf.greater(K_N, 0.5), ones, zeros)
			X_N = X_N * K_N

			var_decoder = tf.get_variable('var_decoder', shape=[12, 304], dtype=tf.float32)
			X_C = tf.gather_nd(var_decoder, X_C)
			

			# concat
			X1 = tf.concat([X_C, X_N, K_N], axis=1)
			# layer 1
			X1 = tf.layers.dense(
				X1, 
				units=2048, 
				kernel_regularizer=kernel_regularizer,
				bias_regularizer=bias_regularizer,
				activation=tf.nn.tanh)
			X1 = tf.layers.dropout(X1, training=drop_hidden, rate=HIDDEN_DROPRATE)
			
			# layer 2
			X2 = tf.layers.dense(
				X1, 
				units=2048, 
				kernel_regularizer=kernel_regularizer,
				bias_regularizer=bias_regularizer,
				activation=tf.nn.tanh)
			X2 = X2 + X1
			X2 = tf.layers.dropout(X2, training=drop_hidden, rate=HIDDEN_DROPRATE)

			# layer 3
			X3 = tf.layers.dense(
				X2, 
				units=1024, 
				kernel_regularizer=kernel_regularizer,
				bias_regularizer=bias_regularizer,
				activation=tf.nn.tanh)
			X3 = tf.layers.dropout(X3, training=drop_hidden, rate=HIDDEN_DROPRATE)

			# layer 4
			X4 = tf.layers.dense(
				X3, 
				units=1024,
				kernel_regularizer=kernel_regularizer,
				bias_regularizer=bias_regularizer)
			X4 = X4 + X3
			# layer 5
			Y1 = tf.layers.dense(
				X4, 
				units=NUM_SCORED_OUTPUT,
				kernel_regularizer=kernel_regularizer,
				bias_regularizer=bias_regularizer)
			Y1 = tf.nn.sigmoid(Y1)

			Y2 = None
			if return_nonscored_output:
				Y2 = tf.layers.dense(
					X4, 
					units=NUM_NONSCORED_OUTPUT,
					kernel_regularizer=kernel_regularizer,
					bias_regularizer=bias_regularizer)
				Y2 = tf.nn.sigmoid(Y2)
			return Y1, Y2

	def compute_cost(self, Y1, Y2, PY1, PY2):
		cost1 = -tf.reduce_mean(Y1*tf.log(PY1+1e-6) + (1-Y1)*tf.log(1+1e-6-PY1))
		cost2 = -tf.reduce_mean(Y2*tf.log(PY2+1e-6) + (1-Y2)*tf.log(1+1e-6-PY2))
		cost = cost1 + cost2 * NONSCORE_WEIGHT
		return cost1, cost2, cost

	def train(self, train_feature_file, scored_target_file, nonscored_target_file, validation_rate=0.2, n_epochs=30, batch_size=100, model_path='./model/model', resume=False):
		X_C = tf.placeholder(tf.int32, shape=[None, NUM_CATEGORICAL_ATTR]) # categorical feature holders
		X_N = tf.placeholder(tf.float32, shape=[None, NUM_NUMERICAL_ATTR]) # numerical feature holders
		Y1 = tf.placeholder(tf.float32, shape=[None, NUM_SCORED_OUTPUT]) # scored targets holder
		Y2 = tf.placeholder(tf.float32, shape=[None, NUM_NONSCORED_OUTPUT]) # nonscored targets holder
		N = tf.placeholder(tf.bool, shape=()) # add noise holder 
		D_I = tf.placeholder(tf.bool, shape=()) # drop_input holder
		D_H = tf.placeholder(tf.bool, shape=()) # drop_hidden holder
		PY1, PY2 = self.build_model(X_C, X_N, add_noise=N, drop_input=D_I, drop_hidden=D_H, return_nonscored_output=True)
		cost1, cost2, cost = self.compute_cost(Y1, Y2, PY1, PY2)
		train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())

		dataset = self.make_dataset(train_feature_file, scored_target_file, nonscored_target_file)
		train_dataset, valid_dataset = self.split_dataset(dataset, test_rate=validation_rate)
		n_train_data = len(train_dataset[0])
		n_test_data = len(valid_dataset[0])
		min_valid_loss = 1.0
		for i in range(n_epochs):
			
			###### train ######			
			_, categorical_features, numerical_features, scored_targets, nonscored_targets = self.shuffle_dataset(train_dataset)
			mean_cost1 = 0
			mean_cost2 = 0
			for j in range(0, n_train_data, batch_size):
				end_j = min(n_train_data, j+batch_size)
				x_c = categorical_features[j:end_j]
				x_n = numerical_features[j:end_j]
				y1 = scored_targets[j:end_j]
				y2 = nonscored_targets[j:end_j]
				py1_val, c_val1, c_val2, _ = session.run([PY1, cost1, cost2, train_op], feed_dict={X_C: x_c, X_N: x_n, Y1: y1, Y2: y2, N: True, D_I: True, D_H: True})
				mean_cost1 = (mean_cost1*j+c_val1*(end_j-j))/end_j
				mean_cost2 = (mean_cost2*j+c_val2*(end_j-j))/end_j
				print('Epoch {:04d}, Progress {:06d}, MCost1 {:05f}, MCost2 {:05f}'.format(i, j, mean_cost1, mean_cost2), end='\r')
			
			###### valid ######
			_, categorical_features, numerical_features, scored_targets, nonscored_targets = valid_dataset
			valid_losses = []
			for j in range(0, n_test_data, batch_size):
				temp_scored_targets = []
				end_j = min(j+batch_size, n_test_data)
				x_c = categorical_features[j:end_j]
				x_n = numerical_features[j:end_j]
				y1 = scored_targets[j:end_j]
				for k in range(50):
					pred_y1 = session.run(PY1, feed_dict={X_C: x_c, X_N: x_n, N: False, D_I: True, D_H: True})
					temp_scored_targets.append(pred_y1)
				temp_scored_targets = np.float32(temp_scored_targets)
				temp_scored_targets = np.mean(temp_scored_targets, axis=0)
				valid_loss = -np.mean((y1*np.log(temp_scored_targets+1e-6)+(1-y1)*np.log(1+1e-6-temp_scored_targets)), axis=1)
				valid_losses.append(valid_loss)
			valid_losses = np.concatenate(valid_losses, axis=0)
			valid_loss = np.mean(valid_losses)
			
			print('Epoch {:04d}, MCost1 {:05f}, MCost2 {:05f}, Valid Loss {:05f}\t\t\t\t\t\t\t'.format(i, mean_cost1, mean_cost2, valid_loss))
			#print('Epoch {:04d}, Valid Loss {:05f}\t\t\t\t\t\t\t'.format(i, valid_loss))
		saver.save(session, model_path)
		session.close()

	def test(self, test_feature_file, sample_submission_file, batch_size=100, model_path='./model/model'):
		X_C = tf.placeholder(tf.int32, shape=[None, NUM_CATEGORICAL_ATTR])
		X_N = tf.placeholder(tf.float32, shape=[None, NUM_NUMERICAL_ATTR])
		N = tf.placeholder(tf.bool, shape=()) # add noise holder 
		D_I = tf.placeholder(tf.bool, shape=()) # drop_input holder
		D_H = tf.placeholder(tf.bool, shape=()) # drop_hidden holder
		B = tf.placeholder(tf.bool, shape=()) # use batch_normalization in training node
		PY1, _ = self.build_model(X_C, X_N, add_noise=N, drop_input=D_I, drop_hidden=D_H, batch_norm=B, return_nonscored_output=False)
		
		session = tf.Session()
		saver = tf.train.Saver()
		saver.restore(session, model_path)
		indices, categorical_features, numerical_features, _, _ = self.make_dataset(test_feature_file, None, None)
		n_data = len(categorical_features)
		pred_targets = []
		for i in range(0, n_data, batch_size):
			end_i = min(i+batch_size, n_data)
			x_c = categorical_features[i:end_i]
			x_n = numerical_features[i:end_i]
			temp_pred_targets = []
			for k in range(NUM_DECISIONS):
				pred = session.run(PY1, feed_dict={X_C: x_c, X_N: x_n, N: False, D_I: True, D_H: True, B: False})
				temp_pred_targets.append(pred)
			temp_pred_targets = np.float32(temp_pred_targets)
			temp_pred_targets = np.mean(temp_pred_targets, axis=0)
			pred_targets.append(pred) # TODO: run 50 times
		session.close()
		pred_targets = np.concatenate(pred_targets, axis=0)
		dt = pd.read_csv(sample_submission_file)
		columns = dt.columns.values
		for i, column in enumerate(columns[1:]):
			dt[column] = pred_targets[:,i]
		dt.to_csv('/kaggle/working/submission.csv', index = False)
		print('Done')
    
	def train_and_test(self, train_feature_file, scored_target_file, nonscored_target_file, test_feature_file, sample_submission_file, n_epochs=30, batch_size=100, model_path='./model/model'):
		X_C = tf.placeholder(tf.int32, shape=[None, NUM_CATEGORICAL_ATTR]) # categorical feature holders
		X_N = tf.placeholder(tf.float32, shape=[None, NUM_NUMERICAL_ATTR]) # numerical feature holders
		Y1 = tf.placeholder(tf.float32, shape=[None, NUM_SCORED_OUTPUT]) # scored targets holder
		Y2 = tf.placeholder(tf.float32, shape=[None, NUM_NONSCORED_OUTPUT]) # nonscored targets holder
		N = tf.placeholder(tf.bool, shape=()) # add noise holder 
		D_I = tf.placeholder(tf.bool, shape=()) # drop_input holder
		D_H = tf.placeholder(tf.bool, shape=()) # drop_hidden holder
		PY1, PY2 = self.build_model(X_C, X_N, add_noise=N, drop_input=D_I, drop_hidden=D_H, return_nonscored_output=True)
		
		cost1, cost2, cost = self.compute_cost(Y1, Y2, PY1, PY2)
		train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
		session = tf.Session()
		saver = tf.train.Saver()
		global_init = tf.global_variables_initializer()

		dataset = self.make_dataset(train_feature_file, scored_target_file, nonscored_target_file)
		test_dataset = self.make_dataset(test_feature_file, None, None)
		full_pred_targets = 0
		for n in range(10):
			tf.set_random_seed(n)
			session.run(global_init)
			train_dataset, valid_dataset = self.split_dataset(dataset, test_rate=0.2)
			############################### train ###############################
			n_train_data = len(train_dataset[0])
			for i in range(n_epochs):			
				###### train ######			
				_, categorical_features, numerical_features, scored_targets, nonscored_targets = self.shuffle_dataset(train_dataset)
				#_, categorical_features, numerical_features, scored_targets, nonscored_targets = train_dataset
				mean_cost1 = 0
				mean_cost2 = 0
				for j in range(0, n_train_data, batch_size):
					end_j = min(n_train_data, j+batch_size)
					x_c = categorical_features[j:end_j]
					x_n = numerical_features[j:end_j]
					y1 = scored_targets[j:end_j]
					y2 = nonscored_targets[j:end_j]
					py1_val, c_val1, c_val2, _ = session.run([PY1, cost1, cost2, train_op], feed_dict={X_C: x_c, X_N: x_n, Y1: y1, Y2: y2, N: True, D_I: True, D_H: True})
					mean_cost1 = (mean_cost1*j+c_val1*(end_j-j))/end_j
					mean_cost2 = (mean_cost2*j+c_val2*(end_j-j))/end_j
					print('Epoch {:04d}, Progress {:06d}, MCost1 {:05f}, MCost2 {:05f}'.format(i, j, mean_cost1, mean_cost2), end='\r')			
				
				###### valid ######
				_, categorical_features, numerical_features, scored_targets, nonscored_targets = valid_dataset
				n_valid_data = len(categorical_features)
				valid_losses = []
				for j in range(0, n_valid_data, batch_size):
					temp_scored_targets = []
					end_j = min(j+batch_size, n_valid_data)
					x_c = categorical_features[j:end_j]
					x_n = numerical_features[j:end_j]
					y1 = scored_targets[j:end_j]
					for k in range(1):
						pred_y1 = session.run(PY1, feed_dict={X_C: x_c, X_N: x_n, N: False, D_I: True, D_H: True})
						temp_scored_targets.append(pred_y1)
					temp_scored_targets = np.float32(temp_scored_targets)
					temp_scored_targets = np.mean(temp_scored_targets, axis=0)
					valid_loss = -np.mean((y1*np.log(temp_scored_targets+1e-6)+(1-y1)*np.log(1+1e-6-temp_scored_targets)), axis=1)
					valid_losses.append(valid_loss)
				valid_losses = np.concatenate(valid_losses, axis=0)
				valid_loss = np.mean(valid_losses)
				print('Epoch {:04d}, MCost1 {:05f}, MCost2 {:05f}, Valid Loss {:05f}\t\t\t\t\t\t\t'.format(i, mean_cost1, mean_cost2, valid_loss))

			############################### test ###############################
			indices, categorical_features, numerical_features, _, _ = test_dataset
			n_test_data = len(categorical_features)
			pred_targets = []
			for i in range(0, n_test_data, batch_size):
				end_i = min(i+batch_size, n_test_data)
				x_c = categorical_features[i:end_i]
				x_n = numerical_features[i:end_i]
				temp_pred_targets = []
				for k in range(NUM_DECISIONS):
					pred = session.run(PY1, feed_dict={X_C: x_c, X_N: x_n, N: False, D_I: True, D_H: True})
					temp_pred_targets.append(pred)
				temp_pred_targets = np.float32(temp_pred_targets)
				temp_pred_targets = np.mean(temp_pred_targets, axis=0)
				pred_targets.append(temp_pred_targets)
			pred_targets = np.concatenate(pred_targets, axis=0)
			full_pred_targets = (full_pred_targets * n + pred_targets)/(n+1)

		dt = pd.read_csv(sample_submission_file)
		dt['sig_id'] = indices
		columns = dt.columns.values
		for i, column in enumerate(columns[1:]):
			dt[column] = full_pred_targets[:,i]
		dt.to_csv('submission.csv', index = False)
		print('Done')
		session.close()



model = Model()
base_folder = './'
output_folder = './'
#base_folder = '/kaggle/input/lish-moa/'
#output_folder = '/kaggle/working/'
'''
model.train(
	train_feature_file = base_folder + 'train_features.csv', 
	scored_target_file = base_folder + 'train_targets_scored.csv', 
	nonscored_target_file = base_folder + 'train_targets_nonscored.csv',
	n_epochs=60, 
	batch_size=100, 
	model_path=output_folder + 'model/model', 
	resume=RESUME)
'''
model.train_and_test(
	train_feature_file = base_folder + 'train_features.csv', 
	scored_target_file = base_folder + 'train_targets_scored.csv', 
	nonscored_target_file = base_folder + 'train_targets_nonscored.csv',
	test_feature_file = base_folder + 'test_features.csv', 	
	sample_submission_file=base_folder + 'sample_submission.csv',
	n_epochs=100, 
	batch_size=100, 
	model_path=output_folder + 'model/model')
