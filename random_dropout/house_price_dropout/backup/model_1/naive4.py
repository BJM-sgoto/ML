import pandas as pd
import numpy as np
import datetime
import re
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()
#np.random.seed(1)
#tf.set_random_seed(1)

TRAIN_BATCH_SIZE = 100
class Model:
	def __init__(self, train_file_path='train.csv', categorical_attribute_file='categorical_attributes.txt', numerical_attribute_file='numerical_attributes.txt', read_file=False):
		# constants
		self.max_droprate = {
			'MSSubClass': 0.1,
			'MSZoning': 0.2,
			'LotFrontage': 0.2,
			'LotArea': 0.1,
			'Street':  0.3,
			'Alley': 0.3,
			'LotShape': 0.1,
			'LandContour':  0.2,
			'Utilities': 0.2,
			'LotConfig': 0.1,
			'LandSlope': 0.2,
			'Neighborhood':  0.2,
			'Condition1': 0.3,
			'BldgType': 0.2,
			'HouseStyle': 0.2,
			'OverallQual': 0.2,
			'OverallCond': 0.1,
			'YearBuilt': 0.2,
			'YearRemodAdd': 0.2,
			'Functional':	0.1,

			'RoofStyle': 0.3,
			'RoofMatl': 0.3,
			'Exterior1st': 0.3,
			'Exterior2nd': 0.3,
			'MasVnrType': 0.3,
			'MasVnrArea': 0.3,
			'ExterQual': 0.3,
			'ExterCond': 0.3,

			'Foundation': 0.1,
			'BsmtQual': 0.2,
			'BsmtCond': 0.2,
			'BsmtExposure': 0.2,
			'BsmtFinType1': 0.2,
			'BsmtFinSF1': 0.2,
			'BsmtFinType2': 0.2,
			'BsmtFinSF2': 0.2,
			'BsmtUnfSF': 0.2,
			'Heating': 0.3,
			'HeatingQC': 0.2,
			'CentralAir': 0.2,
			'Electrical': 0.2,
			'1stFlrSF': 0.1,
			'2ndFlrSF': 0.1,
			'LowQualFinSF': 0.2,
			'GrLivArea': 0.1,
			'BsmtFullBath': 0.2,
			'BsmtHalfBath': 0.2,
			'FullBath': 0.2,
			'HalfBath': 0.2,
			'BedroomAbvGr': 0.2,
			'KitchenAbvGr': 0.2,
			'KitchenQual': 0.2,
			'Fireplaces': 0.3,
			'FireplaceQu':0.3,
			'EnclosedPorch': 0.3,
			'3SsnPorch': 0.1,
			'ScreenPorch': 0.2,

			'GarageType': 0.2,
			'GarageYrBlt': 0.2,
			'GarageFinish': 0.2,
			'GarageCars': 0.2,
			'GarageArea': 0.2,
			'GarageQual': 0.2,
			'GarageCond': 0.2,
			'PavedDrive': 0.3,
			'WoodDeckSF': 0.3,
			'OpenPorchSF': 0.3,
			'PoolArea': 0.1,
			'PoolQC': 0.1,
			'Fence': 0.3,

			'MiscVal': 0.001,
			'YrSold': 0.2,
			'SaleType': 0.2,
			'SaleCondition': 0.2
			}
		start = datetime.datetime.now()
		_, raw_dataset = self.make_raw_dataset(train_file_path)
		end = datetime.datetime.now()
		print('Make raw dataset in', end - start)
		start = end
		if not read_file:
			self.categorical_attributes = self.write_categorical_values(raw_dataset, categorical_attribute_file)
			self.numerical_attributes = self.write_numerical_attributes(raw_dataset, numerical_attribute_file)
		else:
			self.categorical_attributes = self.read_categorical_attributes(categorical_attribute_file)
			self.numerical_attributes = self.read_numerical_attributes(numerical_attribute_file)
		end = datetime.datetime.now()
		print('Make categorical, numerical attributes in', end - start)
		start = end
		self.dataset, self.nan_mask = self.transform_dataset(raw_dataset)
		end = datetime.datetime.now()
		print('Transform dataset in', end - start)

	def replace_NA(self, raw_dataset):
		# replace NA with meaningful names
		replace_attrs = [
			{'attri_name': 'Alley', 'replace_NA_with': 'No_Alley'},
			{'attri_name': 'BsmtQual', 'replace_NA_with': 'No_Basement'},
			{'attri_name': 'BsmtCond', 'replace_NA_with': 'No_Basement'},
			{'attri_name': 'BsmtExposure', 'replace_NA_with': 'No_Basement'},
			{'attri_name': 'BsmtFinType1', 'replace_NA_with': 'No_Basement'},
			{'attri_name': 'BsmtFinType2', 'replace_NA_with': 'No_Basement'},
			{'attri_name': 'FireplaceQu', 'replace_NA_with': 'No_Fireplace'},
			{'attri_name': 'GarageType', 'replace_NA_with': 'No_Garage'},
			{'attri_name': 'GarageYrBlt', 'replace_NA_with': 'No_Garage'},
			{'attri_name': 'GarageFinish', 'replace_NA_with': 'No_Garage'},
			{'attri_name': 'GarageArea', 'replace_NA_with': 'No_Garage'},
			{'attri_name': 'GarageQual', 'replace_NA_with': 'No_Garage'},
			{'attri_name': 'GarageCond', 'replace_NA_with': 'No_Garage'},
			{'attri_name': 'PoolQC', 'replace_NA_with': 'No_Pool'},
			{'attri_name': 'Fence', 'replace_NA_with': 'No_Fence'}
		]
		# replace
		for attr in replace_attrs:
			raw_dataset[attr['attri_name']] = raw_dataset[attr['attri_name']].replace('nan', attr['replace_NA_with'])
		return raw_dataset
		
	def make_raw_dataset(self, file_path):
		dataset = pd.read_csv(file_path)
		# remove meaningless attributes	
		unimportant_attrs = ['Id', 'Condition2', 'TotalBsmtSF', 'TotRmsAbvGrd', 'MiscFeature', 'MoSold']
		index = dataset['Id']
		for unimportant_attr in unimportant_attrs:
			dataset = dataset.drop(unimportant_attr, axis=1)
		# convert to string
		columns = dataset.columns.values
		for column in columns:
			dataset[column] = dataset[column].astype('str')
		
		dataset = self.replace_NA(dataset)
		
		return index, dataset
	
	def write_categorical_values(self, raw_dataset, output_file):
		# categorical attributes are attributes with less than 30 unique values
		dataset = raw_dataset
		categorical_attribute_names = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'Functional', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
		columns = dataset.columns.values
		categorical_attributes = {}
		f = open(output_file, 'w')
		for categorical_attribute in categorical_attribute_names:
			if categorical_attribute in columns:
				unique_values = list(dataset[categorical_attribute].unique())
				if len(unique_values)<30:
					if 'nan' in unique_values:
						unique_values.remove('nan')
					unique_values.sort()
					f.write(categorical_attribute + '\t' + str(unique_values) + '\n')
					categorical_attributes[categorical_attribute] = unique_values
		f.close()
		return categorical_attributes
	
	def read_categorical_attributes(self, input_file):	
		f = open(input_file, 'r')
		s = f.read().strip()
		f.close()
		attributes = s.split('\n')
		categorical_attributes = {}
		for attribute in attributes:
			attribute_name, unique_values = attribute.split('\t')
			categorical_attributes[attribute_name] = eval(unique_values)
		return categorical_attributes
	
	def write_numerical_attributes(self, raw_dataset, output_file):
		dataset = raw_dataset
		categorical_attribute_names = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'Functional', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
		columns = dataset.columns.values
		numerical_attributes = {}
		f = open(output_file, 'w')
		for column in columns:
			if column not in categorical_attribute_names and column!='SalePrice':
				numerical_attribute = column
				minval = 10000000
				maxval = -10000000
				unique_values = list(dataset[numerical_attribute].unique())
				for unique_value in unique_values:
					if unique_value!='nan' and not unique_value.startswith('No'):
						temp = float(unique_value)
						#if temp>0.001:
						if temp<minval:
							minval = temp
						elif temp>maxval:
							maxval = temp
				numerical_attributes[numerical_attribute] = {'minval': minval, 'maxval': maxval}
				f.write(numerical_attribute + '\t' + str(minval) + '\t' + str(maxval) + '\n')

		# output numerical attributes
		output_data = np.float32(dataset['SalePrice'].values)
		mean, std = self._make_output_param(output_data)
		numerical_attributes['SalePrice'] = {'mean': mean, 'std': std}
		f.write('SalePrice\t' + str(mean) + '\t' + str(std) + '\n')

		f.close()
		return numerical_attributes

	def _make_output_param(self, data):
		mean = np.mean(data)
		std = np.std(data)
		return mean, std
	
	def read_numerical_attributes(self, input_file):
		f = open(input_file, 'r')
		s = f.readline().strip()
		numerical_attributes = {}
		while s:
			elements = s.split('\t')
			if elements[0]!='SalePrice':
				numerical_attributes[elements[0]] = {'minval': float(elements[1]), 'maxval': float(elements[2])}
			else:
				numerical_attributes[elements[0]] = {'mean': float(elements[1]), 'std': float(elements[2])}
			s = f.readline().strip()
		f.close()
		return numerical_attributes
	
	def transform_dataset(self, raw_dataset):
		dataset = raw_dataset
		columns = dataset.columns.values
		categorical_attribute_names = list(self.categorical_attributes.keys())
		processed_dataset = {}
		nan_mask = {}
		for column in columns:
			processed_dataset[column] = []
			nan_mask[column] = []
			
		for i, row in dataset.iterrows():
			for column in columns:
				if column in categorical_attribute_names: # if categorical attribute
					unique_values = self.categorical_attributes[column]
					if row[column] in unique_values:
						processed_dataset[column].append(unique_values.index(row[column]))
						nan_mask[column].append(0)
					else: # nan
						processed_dataset[column].append(0)
						nan_mask[column].append(1)
				elif column!='SalePrice': # if numerical attribute
					if row[column]!='nan' and not row[column].startswith('No'):
						val = float(row[column])
						#if val>0.001:
						val = (val - self.numerical_attributes[column]['minval'])/(self.numerical_attributes[column]['maxval'] - self.numerical_attributes[column]['minval'])
						processed_dataset[column].append(val)
						nan_mask[column].append(0)
						#else: # nan
						#	processed_dataset[column].append(0)
						#	nan_mask[column].append(1)
					else: # nan
						processed_dataset[column].append(0)
						nan_mask[column].append(1)
				else:
					val = float(row[column])
					processed_dataset[column].append((val - self.numerical_attributes[column]['mean'])/self.numerical_attributes[column]['std'])
					nan_mask[column].append(0)
			
		return processed_dataset, nan_mask
	
	def sample(self, dataset, nan_mask, n_samples=10):
		keys = list(dataset.keys())
		n_data = len(dataset[keys[0]])
		random_sample_ids = np.random.choice(n_data, n_samples)
		categorical_attribute_names = list(self.categorical_attributes.keys())
		batch_input = {}
		batch_output = []
		batch_nan = {}

		# make input and nan_mask
		attribute_names = list(self.categorical_attributes.keys()) + list(self.numerical_attributes.keys())
		attribute_names.remove('SalePrice')
		for attribute_name in attribute_names:
			batch_input[attribute_name] = [dataset[attribute_name][random_id] for random_id in random_sample_ids]
			if attribute_name in categorical_attribute_names:
				batch_input[attribute_name] = np.int32(batch_input[attribute_name])
			else:
				batch_input[attribute_name] = np.float32(batch_input[attribute_name])
			batch_nan[attribute_name] = np.int32([nan_mask[attribute_name][random_id] for random_id in random_sample_ids])
			batch_nan[attribute_name] += np.int32(np.random.rand(n_samples)<self.max_droprate[attribute_name]*np.random.rand()*0.5)*2

		
		n_attributes = len(attribute_names)
		for j in range(n_samples):
			n_nan = 0
			for attribute_name in attribute_names:
				if batch_input[attribute_name][j]!=0:
					n_nan+=1
			# drop too many attributes, find dropped attribute to recover			
			if n_nan==n_attributes:
				random_id = np.random.randint(n_attributes)
				random_attribute = attribute_names[random_id]
				while batch_nan[random_attribute][j]>=2:
					random_id = np.random.randint(n_group_attributes)
					random_attribute = group_attribute_names[random_id]
				batch_nan[random_attribute][j] = 0
			# recover dropped attributes
			for attribute_name in attribute_names:
				if batch_nan[attribute_name][j]>1:
					batch_nan[attribute_name][j]=1
				if batch_nan[attribute_name][j]>0:
					batch_input[attribute_name][j] = 0
		
		# make output
		batch_output = np.float32([dataset['SalePrice'][random_id] for random_id in random_sample_ids])
		return batch_input, batch_nan, batch_output
	
	def build_model(self, Xs, dropout_rates, training=False):
		
		categorical_attribute_names = list(self.categorical_attributes.keys())
		processed_Xs = {}
		keys = list(Xs.keys())
		for key in keys:
			X = Xs[key]['holder']
			X = tf.expand_dims(X, axis=1)
			NAN = Xs[key]['is_nan']
			NAN = tf.expand_dims(NAN, axis=1)
			inverse_NAN = 1 - NAN
			if key in categorical_attribute_names:
				# encode 
				X_EncodeDict = tf.get_variable(name = key + 'EncodeDict', shape = [len(self.categorical_attributes[key]), 10], dtype=tf.float32)
				X = tf.gather_nd(X_EncodeDict, X)
			processed_Xs[key] = {'holder': X, 'weight': inverse_NAN}
		
		
		attribute_names = list(self.categorical_attributes.keys()) + list(self.numerical_attributes.keys())
		attribute_names.remove('SalePrice')
		# first layers
		layer1 = []
		for key in attribute_names:
			print(key)
			layer1.append(processed_Xs[key]['holder'])
			layer1.append(processed_Xs[key]['weight'])

		layer1 = tf.concat(layer1, axis=1)
		layer1 = tf.layers.dense(layer1, units=256, activation=tf.nn.leaky_relu)


		# second layer
		layer2 = tf.layers.dropout(layer1, rate=dropout_rates[0], training=training)
		layer2 = tf.layers.dense(layer2, units=256, activation=tf.nn.leaky_relu)

		# third layer
		layer3 = tf.layers.dropout(layer2, rate=dropout_rates[1], training=training)
		layer3 = tf.layers.dense(layer3, units=128, activation=tf.nn.leaky_relu)

		# forth layer
		layer4 = tf.layers.dropout(layer3, rate=dropout_rates[2], training=training)
		layer4 = tf.layers.dense(layer4, units=64)
		
		# fifth layer
		output = tf.layers.dense(layer4, units=1)
		output = tf.squeeze(output, axis=1)
		return output

	def _make_holder(self, training=False):
		# group 1
		X_MSSubClass, NA_MSSubClass = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_MSZoning, NA_MSZoning = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_LotFrontage, NA_LotFrontage = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_LotArea, NA_LotArea = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Street, NA_Street = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Alley, NA_Alley = tf.placeholder(tf.int32, shape=[None]),tf.placeholder(tf.float32, shape=[None])
		X_LotShape, NA_LotShape = tf.placeholder(tf.int32, shape=[None]),tf.placeholder(tf.float32, shape=[None])
		X_LandContour, NA_LandContour = tf.placeholder(tf.int32, shape=[None]),tf.placeholder(tf.float32, shape=[None])
		X_Utilities, NA_Utilities = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_LotConfig, NA_LotConfig = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_LandSlope, NA_LandSlope = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Neighborhood, NA_Neighborhood = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Condition1, NA_Condition1 = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BldgType, NA_BldgType = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_HouseStyle, NA_HouseStyle = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_OverallQual, NA_OverallQual = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_OverallCond, NA_OverallCond = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_YearBuilt, NA_YearBuilt = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_YearRemodAdd, NA_YearRemodAdd = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Functional, NA_Functional = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		
		# group 2
		X_RoofStyle, NA_RoofStyle = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_RoofMatl, NA_RoofMatl = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Exterior1st, NA_Exterior1st = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Exterior2nd, NA_Exterior2nd = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_MasVnrType, NA_MasVnrType = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_MasVnrArea, NA_MasVnrArea = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_ExterQual, NA_ExterQual = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_ExterCond, NA_ExterCond = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])

		# group 3
		X_Foundation, NA_Foundation = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtQual, NA_BsmtQual = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtCond, NA_BsmtCond = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtExposure, NA_BsmtExposure = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtFinType1, NA_BsmtFinType1 = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtFinSF1, NA_BsmtFinSF1 = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtFinType2, NA_BsmtFinType2 = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtFinSF2, NA_BsmtFinSF2 = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtUnfSF, NA_BsmtUnfSF = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])

		# group 4
		X_Heating, NA_Heating = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_HeatingQC, NA_HeatingQC = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_CentralAir,	NA_CentralAir = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Electrical, NA_Electrical = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_1stFlrSF, NA_1stFlrSF = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_2ndFlrSF, NA_2ndFlrSF = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_LowQualFinSF, NA_LowQualFinSF = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_GrLivArea, NA_GrLivArea = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])

		# group 5
		X_BsmtFullBath, NA_BsmtFullBath = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BsmtHalfBath, NA_BsmtHalfBath = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_FullBath, NA_FullBath = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_HalfBath, NA_HalfBath = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_BedroomAbvGr, NA_BedroomAbvGr = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_KitchenAbvGr, NA_KitchenAbvGr = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_KitchenQual, NA_KitchenQual = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Fireplaces, NA_Fireplaces = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_FireplaceQu, NA_FireplaceQu = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_EnclosedPorch, NA_EnclosedPorch = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_3SsnPorch, NA_3SsnPorch = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_ScreenPorch, NA_ScreenPorch = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])

		# group 6
		X_GarageType, NA_GarageType = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_GarageYrBlt, NA_GarageYrBlt = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_GarageFinish, NA_GarageFinish = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_GarageCars, NA_GarageCars = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_GarageArea, NA_GarageArea = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_GarageQual, NA_GarageQual = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_GarageCond, NA_GarageCond = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_PavedDrive, NA_PavedDrive = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_WoodDeckSF, NA_WoodDeckSF = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_OpenPorchSF, NA_OpenPorchSF = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_PoolArea, NA_PoolArea = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_PoolQC, NA_PoolQC = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_Fence, NA_Fence = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])

		# group 7
		X_MiscVal, NA_MiscVal = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_YrSold, NA_YrSold = tf.placeholder(tf.float32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_SaleType, NA_SaleType = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])
		X_SaleCondition, NA_SaleCondition = tf.placeholder(tf.int32, shape=[None]), tf.placeholder(tf.float32, shape=[None])

		
		Xs = {
		# group 1
		'MSSubClass': {'holder': X_MSSubClass, 'is_nan': NA_MSSubClass},
		'MSZoning': {'holder': X_MSZoning, 'is_nan': NA_MSZoning},
		'LotFrontage': {'holder': X_LotFrontage, 'is_nan': NA_MSZoning},
		'LotArea': {'holder': X_LotFrontage, 'is_nan': NA_LotArea},
		'Street': {'holder': X_Street, 'is_nan': NA_Street},
		'Alley': {'holder': X_Alley, 'is_nan': NA_Alley},
		'LotShape': {'holder': X_LotShape, 'is_nan': NA_LotShape},
		'LandContour': {'holder': X_LandContour, 'is_nan': NA_LandContour},
		'Utilities': {'holder': X_Utilities, 'is_nan': NA_Utilities},
		'LotConfig': {'holder': X_LotConfig, 'is_nan': NA_LotConfig},
		'LandSlope': {'holder': X_LandSlope, 'is_nan': NA_LandSlope},
		'Neighborhood': {'holder': X_Neighborhood, 'is_nan': NA_Neighborhood},
		'Condition1': {'holder': X_Condition1, 'is_nan': NA_Condition1},
		'BldgType': {'holder': X_BldgType, 'is_nan': NA_BldgType},
		'HouseStyle': {'holder': X_HouseStyle, 'is_nan': NA_HouseStyle},
		'OverallQual': {'holder': X_OverallQual, 'is_nan': NA_OverallQual},
		'OverallCond': {'holder': X_OverallCond, 'is_nan': NA_OverallCond},
		'YearBuilt': {'holder': X_YearBuilt, 'is_nan': NA_YearBuilt},
		'YearRemodAdd': {'holder': X_YearRemodAdd, 'is_nan': NA_YearRemodAdd},
		'Functional': {'holder': X_Functional, 'is_nan': NA_Functional},
		# group 2
		'RoofStyle': {'holder': X_RoofStyle, 'is_nan': NA_RoofStyle},
		'RoofMatl': {'holder': X_RoofMatl, 'is_nan': NA_RoofMatl},
		'Exterior1st': {'holder': X_Exterior1st, 'is_nan': NA_Exterior1st},
		'Exterior2nd': {'holder': X_Exterior2nd, 'is_nan': NA_Exterior2nd},
		'MasVnrType': {'holder': X_MasVnrType, 'is_nan': NA_MasVnrType},
		'MasVnrArea': {'holder': X_MasVnrArea, 'is_nan': NA_MasVnrArea},
		'ExterQual': {'holder': X_ExterQual, 'is_nan': NA_ExterQual},
		'ExterCond': {'holder': X_ExterCond, 'is_nan': NA_ExterCond},
		# group 3
		'Foundation': {'holder': X_Foundation, 'is_nan': NA_Foundation},
		'BsmtQual': {'holder': X_BsmtQual, 'is_nan': NA_BsmtQual},
		'BsmtCond': {'holder': X_BsmtCond, 'is_nan': NA_BsmtCond},
		'BsmtExposure': {'holder': X_BsmtExposure, 'is_nan': NA_BsmtExposure},
		'BsmtFinType1': {'holder': X_BsmtFinType1, 'is_nan': NA_BsmtFinType1},
		'BsmtFinSF1': {'holder': X_BsmtFinSF1, 'is_nan': NA_BsmtFinSF1},
		'BsmtFinType2': {'holder': X_BsmtFinType2, 'is_nan': NA_BsmtFinType2},
		'BsmtFinSF2': {'holder': X_BsmtFinSF2, 'is_nan': NA_BsmtFinSF2},
		'BsmtUnfSF': {'holder': X_BsmtUnfSF, 'is_nan': NA_BsmtUnfSF},
		# group 4
		'Heating': {'holder': X_Heating, 'is_nan': NA_Heating},
		'HeatingQC': {'holder': X_HeatingQC, 'is_nan': NA_HeatingQC},
		'CentralAir': {'holder': X_CentralAir, 'is_nan': NA_CentralAir},
		'Electrical': {'holder': X_Electrical, 'is_nan': NA_Electrical},
		'1stFlrSF': {'holder': X_1stFlrSF, 'is_nan': NA_1stFlrSF},
		'2ndFlrSF': {'holder': X_2ndFlrSF, 'is_nan': NA_2ndFlrSF},
		'LowQualFinSF': {'holder': X_LowQualFinSF, 'is_nan': NA_LowQualFinSF},
		'GrLivArea': {'holder': X_GrLivArea, 'is_nan': NA_GrLivArea},
		# group 5
		'BsmtFullBath': {'holder': X_BsmtFullBath, 'is_nan': NA_BsmtFullBath},
		'BsmtHalfBath': {'holder': X_BsmtHalfBath, 'is_nan': NA_BsmtHalfBath},
		'FullBath': {'holder': X_FullBath, 'is_nan': NA_FullBath},
		'HalfBath': {'holder': X_HalfBath, 'is_nan': NA_HalfBath},
		'BedroomAbvGr': {'holder': X_BedroomAbvGr, 'is_nan': NA_BedroomAbvGr},
		'KitchenAbvGr': {'holder': X_KitchenAbvGr, 'is_nan': NA_KitchenAbvGr},
		'KitchenQual': {'holder': X_KitchenQual, 'is_nan': NA_KitchenQual},
		'Fireplaces': {'holder': X_Fireplaces, 'is_nan': NA_Fireplaces},
		'FireplaceQu': {'holder': X_FireplaceQu, 'is_nan': NA_FireplaceQu},
		'EnclosedPorch': {'holder': X_EnclosedPorch, 'is_nan': NA_EnclosedPorch},
		'3SsnPorch': {'holder': X_3SsnPorch, 'is_nan': NA_3SsnPorch},
		'ScreenPorch': {'holder': X_ScreenPorch, 'is_nan': NA_ScreenPorch},
		# group 6
		'GarageType': {'holder': X_GarageType, 'is_nan': NA_GarageType},
		'GarageYrBlt': {'holder': X_GarageYrBlt, 'is_nan': NA_GarageYrBlt},
		'GarageFinish': {'holder': X_GarageFinish, 'is_nan': NA_GarageFinish},
		'GarageCars': {'holder': X_GarageCars, 'is_nan': NA_GarageCars},
		'GarageArea': {'holder': X_GarageArea, 'is_nan': NA_GarageArea},
		'GarageQual': {'holder': X_GarageQual, 'is_nan': NA_GarageQual},
		'GarageCond': {'holder': X_GarageCond, 'is_nan': NA_GarageCond},
		'PavedDrive': {'holder': X_PavedDrive, 'is_nan': NA_PavedDrive},
		'WoodDeckSF': {'holder': X_WoodDeckSF, 'is_nan': NA_WoodDeckSF},
		'OpenPorchSF': {'holder': X_OpenPorchSF, 'is_nan': NA_OpenPorchSF},
		'PoolArea': {'holder': X_PoolArea, 'is_nan': NA_PoolArea},
		'PoolQC': {'holder': X_PoolQC, 'is_nan': NA_PoolQC},
		'Fence': {'holder': X_Fence, 'is_nan': NA_Fence},

		# group 7
		'MiscVal': {'holder': X_MiscVal, 'is_nan': NA_MiscVal},
		'YrSold': {'holder': X_YrSold, 'is_nan': NA_YrSold},
		'SaleType': {'holder': X_SaleType, 'is_nan': NA_SaleType},
		'SaleCondition': {'holder': X_SaleCondition, 'is_nan': NA_SaleCondition},
		}
		# drop rate
		DropRate1 = tf.placeholder(tf.float32, shape=())
		DropRate2 = tf.placeholder(tf.float32, shape=())
		DropRate3 = tf.placeholder(tf.float32, shape=())
		DropRates = [DropRate1, DropRate2, DropRate3]
		if training:			
			# output
			Y = tf.placeholder(tf.float32, shape=[None])
			#
			return Xs, Y, DropRates
		return Xs, DropRates

	def train(self, num_steps=1000, model_path='./model/model', resume=False):
		Xs, Y, DropRates = self._make_holder(training=True)
		PY = self.build_model(Xs, dropout_rates=DropRates, training=True)
		loss = tf.reduce_mean(tf.abs(PY - Y))
		train_op = tf.train.AdamOptimizer().minimize(loss)
		saver = tf.train.Saver()
		session = tf.Session()
		
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		for i in range(num_steps):
			batch_input, batch_nan, batch_output = self.sample(self.dataset, self.nan_mask, n_samples=100)
			dropout_rate1, dropout_rate2, dropout_rate3 = np.random.rand(3)*0.3
			#dropout_rate1, dropout_rate2, dropout_rate3 = 0.5, 0.5, 0.5
			feed_dict = {}
			for key in Xs.keys():
				feed_dict[Xs[key]['holder']] = batch_input[key]
				feed_dict[Xs[key]['is_nan']] = batch_nan[key]
			feed_dict[DropRates[0]] = dropout_rate1
			feed_dict[DropRates[1]] = dropout_rate2
			feed_dict[DropRates[2]] = dropout_rate3
			feed_dict[Y] = batch_output
			loss_val, _ = session.run([loss, train_op], feed_dict=feed_dict)
			print('Step {:04d}, Loss {:06f}'.format(i, loss_val))
		saver.save(session, model_path)
		session.close()

	def test(self, test_file_path, model_path='./model/model'):
		Xs, DropRates = self._make_holder(training=False)
		PY = self.build_model(Xs, dropout_rates=DropRates, training=False)
		saver = tf.train.Saver()
		session = tf.Session()
		saver.restore(session, model_path)
		index, raw_dataset = self.make_raw_dataset(test_file_path)
		dataset, nan_mask = self.transform_dataset(raw_dataset)
		n_data = len(dataset[list(Xs.keys())[0]])
		feed_dict = {}
		categorical_attribute_names = list(self.categorical_attributes.keys())
		output = np.float32([])
		for i in range(0, n_data, 100):
			end_i = min(i+100, n_data)
			batch_input = {}
			batch_output = []
			batch_nan = {}
			attribute_names = list(self.categorical_attributes.keys()) + list(self.numerical_attributes.keys())
			attribute_names.remove('SalePrice')
			for attribute_name in attribute_names:
				batch_input[attribute_name] = [dataset[attribute_name][k] for k in range(i, end_i)]
				if attribute_name in categorical_attribute_names:
					batch_input[attribute_name] = np.int32(batch_input[attribute_name])
				else:
					batch_input[attribute_name] = np.float32(batch_input[attribute_name])
				batch_nan[attribute_name] = np.int32([nan_mask[attribute_name][k] for k in range(i, end_i)])
			feed_dict = {}
			for key in Xs.keys():
				feed_dict[Xs[key]['holder']] = batch_input[key]
				feed_dict[Xs[key]['is_nan']] = batch_nan[key]
			feed_dict[DropRates[0]] = 0.0
			feed_dict[DropRates[1]] = 0.0
			feed_dict[DropRates[2]] = 0.0
			y_val = session.run(PY, feed_dict=feed_dict)
			mean = self.numerical_attributes['SalePrice']['mean']
			std = self.numerical_attributes['SalePrice']['std']
			y_val = mean + y_val*std
			output = np.concatenate([output, y_val])
		ids = index.values
		dataframe = pd.DataFrame({
			'Id': ids,
			'SalePrice': output,
			})
		dataframe.to_csv('submission.csv', index=False)
		session.close()


model = Model(
	train_file_path='train.csv', 
	categorical_attribute_file='categorical_attributes.txt', 
	numerical_attribute_file='numerical_attributes.txt',
	read_file=False)
#model.train(num_steps=3000, model_path='./model/model', resume=False)
model.test(test_file_path='./test.csv', model_path='./model/model')