import tensorflow as tf
import numpy as np

hidden_size = 3
# cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
# cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)

x_data = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(outputs)
