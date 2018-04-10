import tensorflow as tf

x_train = tf.placeholder(tf.float32, shape=(3))
y_train = tf.placeholder(tf.float32, shape=(3))

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = w * x_train + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
# gvs = optimizer.compute_gradients(cost)
# apply_gradients = optimizer.apply_gradients(gvs)
"""
learning_rate = 0.01
gradient = tf.reduce_mean((W * x + b - y) * x)  # gradient of cost
descent = w - learning_rate * gradient
update = w.assign(descent)
"""

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict={x_train: [1, 2, 3], y_train: [2, 4, 6]})
    if (step + 1) % 20 == 0:
        print(step + 1, cost_val, w_val, b_val)
