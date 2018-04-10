import tensorflow as tf

# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0, tf.float32)
# node3 = tf.add(node1, node2)

# sess = tf.Session()
# result = sess.run(node3)
# print(result)

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
node3 = tf.add(node1, node2)

sess = tf.Session()
result = sess.run(node3, feed_dict={node1: [[1, 2], [3, 4]], node2: [[2, 4], [6, 8]]})
print(result)
