#%tensorflow_version 1.x
import tensorflow.compat.v1 as tf
import numpy as np
import os 
import cv2
import random
import datetime

IMG_WIDTH = 224
IMG_HEIGHT = 224

RANDOM_SEED = 1234
tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class CustomModel:
	def __init__(self):
		
		gpus = tf.config.experimental.list_physical_devices('GPU')
		tf.config.experimental.set_virtual_device_configuration(
			gpus[0],
			[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536),
			tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)])
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
		
		self.classes = ['dog', 'cat']
		self.classifier_count = 0
	
	def make_dataset(self, image_folder, class_names):
		sub_folders = os.listdir(image_folder)
		dataset = []
		for i, class_name in enumerate(class_names):
			if class_name in sub_folders:
				folder_path = image_folder + class_name + '/'
				output = np.zeros(len(class_names), dtype=np.float32)
				output[i] = 1
				for file_name in os.listdir(folder_path):
					dataset.append([folder_path + file_name, np.copy(output)])
			else:
				raise Exception('Class {:s} is not in folder {:s}'.format(class_name, image_folder))
		return dataset
	
	def shuffle_dataset(self, dataset):
		random.shuffle(dataset)
	
	def classify(self, images, output=None, training=False):
		with tf.variable_scope('classifier_' + str(self.classifier_count), reuse=False):
			self.classifier_count+=1  
			features = images
			if training:
				noise = tf.random.uniform(shape=tf.shape(features), minval=-5.0, maxval=5.0, dtype=tf.float32)
				features = features + noise
				features = tf.where(tf.greater(features, 255), tf.ones_like(features) * 255, features)
				features = tf.where(tf.less(features, 0), tf.zeros_like(features), features)
			features = features / 255
			layers = [
			{'depth':32},
			{'depth':32},
			{'depth':64},
			{'depth':64},
			{'depth':128}]
			for layer in layers:
				features = tf.layers.conv2d(
					features,
					filters=layer['depth'],
					kernel_size=(3,3),
					strides=(1,1),
					padding='same',
					activation=tf.nn.elu)
				features = tf.layers.max_pooling2d(
					features,
					pool_size=(2,2),
					strides=(2,2))
				features = tf.layers.batch_normalization(
					features,
					training=training)
				
			features = tf.layers.flatten(features)
			features = tf.layers.dense(features, units=64)
			features = tf.layers.batch_normalization(features,training=training)
			features = tf.layers.dense(features, units=len(self.classes))
			predicted_output = tf.nn.softmax(features)
			self.reuse_classifier = True
			if training: # training
				cost = tf.reduce_mean(tf.square(predicted_output - output))
				accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicted_output, axis=1), tf.argmax(output, axis=1)), dtype=tf.float32))
				return predicted_output, cost, accuracy
			else:
				return predicted_output
	
	def combine_model(self, session, gpu0_trainable_variables, gpu1_trainable_variables, gpu0_moving_variables, gpu1_moving_variables, assign_trainable_variables_op, assign_moving_variables_op, trainable_variables_holder, moving_variables_holder):
		gpu_start_time = datetime.datetime.now()
		gpu0_trainable_variables_val = session.run(gpu0_trainable_variables)
		gpu1_trainable_variables_val = session.run(gpu1_trainable_variables)
		gpu0_moving_variables_val = session.run(gpu0_moving_variables)
		gpu1_moving_variables_val = session.run(gpu1_moving_variables)
		gpu_end_time = datetime.datetime.now()
		gpu_time = gpu_end_time - gpu_start_time
		cpu_start_time = datetime.datetime.now()
		num_trainable_variables = len(gpu0_trainable_variables)
		num_moving_variables = len(gpu0_moving_variables)
		feed_dict_trainable_variables = {}
		feed_dict_moving_variables = {}
		for i in range(num_trainable_variables):
			feed_dict_trainable_variables[trainable_variables_holder[i]] = (gpu0_trainable_variables_val[i] + gpu1_trainable_variables_val[i])/2
		for i in range(num_moving_variables):
			feed_dict_moving_variables[moving_variables_holder[i]] = (gpu0_moving_variables_val[i] + gpu1_moving_variables_val[i]) / 2
		cpu_time = datetime.datetime.now() - cpu_start_time
		gpu_start_time = datetime.datetime.now()
		session.run(assign_trainable_variables_op, feed_dict = feed_dict_trainable_variables)
		session.run(assign_moving_variables_op, feed_dict = feed_dict_moving_variables)
		gpu_time += datetime.datetime.now() - gpu_start_time
		return cpu_time, gpu_time
		
	def train_on_batch(self, session, batch, tf_images, tf_outputs, tf_costs, tf_accuracies, tf_train_ops):
		images = []
		outputs = []
		start_cpu_time = datetime.datetime.now()
		for item in batch:
			image = cv2.imread(item[0])
			cut_y, cut_x = np.random.randint(size=[2], low=0, high=21)
			image = image[cut_y: cut_y+IMG_HEIGHT, cut_x: cut_x+IMG_WIDTH]
			images.append(image)
			outputs.append(item[1])
		images = np.float32(images)
		outputs = np.float32(outputs)
		end_cpu_time = datetime.datetime.now()
		cpu_time = end_cpu_time - start_cpu_time
		start_gpu_time = datetime.datetime.now()
		loss_val0, loss_val1, accuracy_val0, accuracy_val1, _, _ = session.run(
			[tf_costs[0], tf_costs[1], tf_accuracies[0], tf_accuracies[1], tf_train_ops[0], tf_train_ops[1]],
			feed_dict={
				tf_images: images,
				tf_outputs: outputs})
		end_gpu_time = datetime.datetime.now()
		gpu_time = end_gpu_time - start_gpu_time
		return (loss_val0 + loss_val1)/2, (accuracy_val0 + accuracy_val1)/2, cpu_time, gpu_time
			
	def train(self, num_epochs=5, batch_size=40, train_folder='./train/', model_path='./multi_gpu/model', resume=False, history_path='./multi_gpu/history.txt'):
		with tf.device('/cpu:0'):
			X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])
			Y = tf.placeholder(tf.float32, shape=[None, len(self.classes)])
			Xs = tf.split(X, 2, axis=0)
			Ys = tf.split(Y, 2, axis=0)
		gpus = ['/gpu:0', '/gpu:1']
		costs = []
		accuracies = []
		train_ops = []
		for i, gpu_id in enumerate(gpus):
			with tf.device(gpu_id):
				PY, cost, accuracy = self.classify(Xs[i], Ys[i], training=True)
		
				cost = tf.reduce_mean(tf.square(PY - Ys[i]))
				accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PY, axis=1), tf.argmax(Ys[i], axis=1)), dtype=tf.float32))
				train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
				update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				train_op = tf.group([train_op, update_op])
				costs.append(cost)
				accuracies.append(accuracy)
				train_ops.append(train_op)
		saver = tf.train.Saver()
		session = tf.Session()
		
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
		gpu0_trainable_variables = tf.trainable_variables('classifier_0')
		gpu0_moving_variables = tf.global_variables('classifier_0[/_a-z0-9]*batch_normalization[/_a-z0-9]*moving')
		gpu1_trainable_variables = tf.trainable_variables('classifier_1')
		gpu1_moving_variables = tf.global_variables('classifier_1[/_a-z0-9]*batch_normalization[/_a-z0-9]*moving')
		trainable_variables_holder = []
		moving_variables_holder = []
		num_trainable_variables = len(gpu0_trainable_variables)
		num_moving_variables = len(gpu0_moving_variables)
		for i in range(num_trainable_variables):
			trainable_variables_holder.append(tf.placeholder(tf.float32, shape=gpu0_trainable_variables[i].get_shape()))
		for i in range(num_moving_variables):
			moving_variables_holder.append(tf.placeholder(tf.float32, shape=gpu0_moving_variables[i].get_shape()))
		
		assign_trainable_variables_op = []
		assign_moving_variables_op = []
		for i in range(num_trainable_variables):
			assign_trainable_variables_op.append(tf.assign(gpu0_trainable_variables[i], trainable_variables_holder[i]))
			assign_trainable_variables_op.append(tf.assign(gpu1_trainable_variables[i], trainable_variables_holder[i]))
		for i in range(num_moving_variables):
			assign_moving_variables_op.append(tf.assign(gpu0_moving_variables[i], moving_variables_holder[i]))
			assign_moving_variables_op.append(tf.assign(gpu1_moving_variables[i], moving_variables_holder[i]))
		
		
		dataset = model.make_dataset(image_folder=train_folder, class_names=self.classes)
		num_data = len(dataset)
		count_to_save = 0
		history_file = open(history_path, 'w')
		history_file.write('Num epochs {:03d}, Batch size {:03d}, Num data {:05d}\n'.format(num_epochs, batch_size, num_data))
		history_file.close()
		mean_loss_val = 0
		mean_accuracy_val = 0
		sum_cpu_time = datetime.timedelta()
		sum_gpu_time = datetime.timedelta()
		for i in range(num_epochs):
			self.shuffle_dataset(dataset)
			for j in range(0, num_data, batch_size):
				end_j = min(num_data, j+batch_size)
				batch = dataset[j: end_j]
				loss_val, accuracy_val, cpu_time, gpu_time = self.train_on_batch(session, batch, X, Y, costs, accuracies, train_ops)
				mean_loss_val = (mean_loss_val*count_to_save+loss_val)/(count_to_save+1)
				mean_accuracy_val = (mean_accuracy_val*count_to_save+accuracy_val)/(count_to_save+1)
				sum_cpu_time+=cpu_time
				sum_gpu_time+=gpu_time
				cpu_time, gpu_time = self.combine_model(session, gpu0_trainable_variables, gpu1_trainable_variables, gpu0_moving_variables, gpu1_moving_variables, assign_trainable_variables_op, assign_moving_variables_op, trainable_variables_holder, moving_variables_holder)
				sum_cpu_time+=cpu_time
				sum_gpu_time+=gpu_time
				count_to_save+=1
				if count_to_save>=50:
					saver.save(session, model_path)
					history_file = open(history_path, 'a')
					history_file.write('Epc {:03d},Prg {:03d},Lss {:05f},Acc {:05f}, MLss {:05f}, MAcc {:05f},CPU {:s},GPU {:s}\n'.format(i, j, loss_val, accuracy_val, mean_loss_val, mean_accuracy_val, str(sum_cpu_time), str(sum_gpu_time)))
					count_to_save = 0
					mean_loss_val = 0
					mean_accuracy_val = 0
				print('Epc {:03d},Prg {:03d},Lss {:05f},Acc {:05f}, MLss {:05f}, MAcc {:05f},CPU {:s},GPU {:s}'.format(i, j, loss_val, accuracy_val, mean_loss_val, mean_accuracy_val, str(sum_cpu_time), str(sum_gpu_time)))
				
		saver.save(session, model_path)
		history_file = open(history_path, 'a')
		history_file.write('Epc {:03d},Prg {:03d},Lss {:05f},Acc {:05f}, MLss {:05f}, MAcc {:05f},CPU {:s},GPU {:s}\n'.format(i, j, loss_val, accuracy_val, mean_loss_val, mean_accuracy_val, str(sum_cpu_time), str(sum_gpu_time)))
		session.close()	
    
#base_folder = '/content/gdrive/My Drive/machine_learning_data/'
base_folder = './'
model = CustomModel()
model.train(
    num_epochs=10, 
    batch_size=40, 
    train_folder = base_folder + 'small_train/', 
    model_path = base_folder + 'multi_gpu/model', 
		history_path='./multi_gpu/history.txt',
    resume=False
)