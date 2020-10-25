import tensorflow as tf
import numpy as np

n_inputs = 4
n_neurons = 6
n_timesteps = 2
X_batch = np.array([
	[[0, 1, 2, 5], [9, 8, 7, 4]], # Batch 1
	[[3, 4, 5, 2], [0, 0, 0, 0]], # Batch 2
	[[6, 7, 8, 5], [6, 5, 4, 2]], # Batch 3
	]) # 3 X 2 X 4 : batch size X timestep X 4 inputs
X = tf.placeholder(tf.float32, [None, n_timesteps, n_inputs])
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
print(outputs, states)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch})
print('outputs_val')
print(outputs_val)
print('Trainable Variables')
w = sess.run(tf.trainable_variables()[0])
x = np.zeros([3, 10])
x[:, 0:4] = X_batch[:,0,:]
x[:, 4:] = np.tanh(np.matmul(x, w))
x[:, 0:4] = X_batch[:,1,:]
print(np.tanh(np.matmul(x,w)))