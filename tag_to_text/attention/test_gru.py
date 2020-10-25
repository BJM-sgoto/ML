import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

ENCODER_DIM = 50

a = tf.placeholder(tf.float32, [None, None, 10])
gru_cell = tf.nn.rnn_cell.GRUCell(ENCODER_DIM)
print(tf.nn.dynamic_rnn(
	gru_cell,
	a,
	dtype=tf.float32,
	time_major=False))