import numpy as np
import tensorflow as tf

class Model:
	def __init__(self):
		self.n_attr = 1433
		self.label_to_classID = {
			'Case_Based': 0,
			'Genetic_Algorithms': 1,
			'Neural_Networks': 2,
			'Probabilistic_Methods': 3,
			'Reinforcement_Learning': 4,
			'Rule_Learning': 5,
			'Theory': 6}
		self.train_rate = 0.8 # use first 80% papers to test
		self.nodes = self.read_nodes() # node structure: (count, feature, label)
		self.edges = self.read_edges()
		#self.full_self_adjacency_matrix = self.make_full_self_adjacency_matrix()
		#self.full_self_adjacency_list = self.make_full_self_adjacency_list()
		print('Finish Initializing')
	
	def read_nodes(self, node_file='./cora/cora.content'):
		f = open(node_file, 'r')
		s = f.readline().strip()
		nodes = {}
		count = 0
		while s:
			elems = s.split('\t')
			nodes[int(elems[0])] = (count, np.float32(elems[1:-1]), self.label_to_classID[elems[-1]])
			count += 1
			s = f.readline().strip()
		f.close()
		return nodes
		
	def make_input_output(self):
		inputs = np.zeros([len(self.nodes), self.n_attr], dtype=np.float32)
		outputs = np.zeros([len(self.nodes)], dtype=np.int32)
		for node_key in self.nodes:
			node_value = self.nodes[node_key]
			node_count = node_value[0]
			inputs[node_count] = node_value[1]
			outputs[node_count] = node_value[2]
		return inputs, outputs
		
	def read_edges(self, edge_file='./cora/cora.cites'):
		f = open(edge_file, 'r')
		s = f.readline().strip()
		edges = {}
		while s:
			elems = s.split('\t')
			start_node = int(elems[1])
			end_node = int(elems[0])
			if not start_node in edges.keys():
				edges[start_node] = [end_node]
			else:
				edges[start_node].append(end_node)
			s = f.readline().strip()
		f.close()
		return edges

	def make_full_self_adjacency_matrix(self):
		n_nodes = len(self.nodes)
		adjacency_matrix = np.zeros([n_nodes, n_nodes])
		for node_id in self.edges:
			start_node_count = self.nodes[node_id][0]
			for end_node in self.edges[node_id]:
				end_node_count = self.nodes[end_node][0]
				adjacency_matrix[start_node_count, end_node_count] = 1
		# make self connection 
		for i in range(n_nodes):
			adjacency_matrix[i,i] = 1
		return adjacency_matrix
	
	def make_full_self_adjacency_list(self):
		n_nodes = len(self.nodes)
		adjacency_list = [[] for i in range(n_nodes)]
		for node_id in self.edges:
			start_node_count = self.nodes[node_id][0]
			for end_node in self.edges[node_id]: 
				end_node_count = self.nodes[end_node][0]
				adjacency_list[start_node_count].append(end_node_count)
		max_len = np.max([len(x) for x in adjacency_list])		
		# make self connection
		for node_count in range(n_nodes):
			adjacency_list[node_count] += [node_count] * (max_len + 1 - len(adjacency_list[node_count]))
		adjacency_list = np.int32(adjacency_list)
		return adjacency_list
	
	def init_model(self, feature_holder, self_adjacency_holder, training=False):
		output_holder = feature_holder
		for i in range(5):
			output_holder = tf.layers.dense(
				output_holder,
				units=64)
			output_holder = tf.gather(
				output_holder,
				self_adjacency_holder,
				axis=0)
			output_holder = tf.layers.dropout(output_holder, rate=0.5, training=training)
			output_holder = tf.reduce_max(output_holder, axis=1)
			output_holder = tf.nn.leaky_relu(output_holder)
		output_holder = tf.layers.dense(
			output_holder,
			units=7)
		return output_holder
	
	def compute_cost(self, labels, logits):
		costs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		n_train_samples = int(len(self.nodes) * 0.8)
		train_cost = tf.reduce_sum(costs[: n_train_samples])/float(n_train_samples)
		return train_cost
		
	def compute_accuracy(self, labels, logits):
		predicted_labels = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
		acc = tf.cast(tf.equal(labels, predicted_labels), dtype=tf.float32)
		n_train_samples = int(len(self.nodes) * 0.8)
		train_acc = tf.reduce_sum(acc[: n_train_samples])/float(n_train_samples)
		test_acc = tf.reduce_sum(acc[n_train_samples:])/float(len(self.nodes) - n_train_samples)
		return train_acc, test_acc
	
	def train(self, n_epochs=1000, model_path='./model/model', resume=False):
		tf.reset_default_graph()
		F = tf.placeholder(tf.float32, [len(self.nodes), self.n_attr])
		N = tf.placeholder(tf.int32, [len(self.nodes), None])
		O = tf.placeholder(tf.int32, [len(self.nodes)])
		PO = self.init_model(F, N, training=False)
		print('Finish Initializing Model', PO)
		
		cost = self.compute_cost(O, PO)
		print('Finish Computing Cost')
		train_acc, test_acc = self.compute_accuracy(O, PO)
		print('Finish Computing Acc')
		
		train_ops = tf.train.AdamOptimizer().minimize(cost)
		print('Finish Making Train Ops')
		session = tf.Session()
		saver = tf.train.Saver()
		if resume:
			saver.restore(session, model_path)
		else:
			session.run(tf.global_variables_initializer())
			
		inputs, outputs = self.make_input_output()
		full_self_adjacency_list = self.make_full_self_adjacency_list()
		feed_dict = {}
		feed_dict[F] = inputs
		feed_dict[O] = outputs
		feed_dict[N] = full_self_adjacency_list
		
		for i in range(n_epochs):
			cost_val, train_acc_val, test_acc_val, _ = session.run([cost, train_acc, test_acc, train_ops], feed_dict=feed_dict)
			print('Progress', i, 'Cost', cost_val, 'Train acc', train_acc_val, 'Test acc val', test_acc_val)
			
		saver.save(session, model_path)
		session.close()
		
		
model = Model()
model.train(
	n_epochs=200, 
	model_path='./model/model', 
	resume=False)