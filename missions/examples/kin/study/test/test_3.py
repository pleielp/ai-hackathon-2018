import tensorflow as tf

# x_1 = tf.placeholder(tf.float32, shape=None)
# x_2 = tf.placeholder(tf.float32, shape=None)
# x_3 = tf.placeholder(tf.float32, shape=None)
x_data = tf.placeholder(tf.float32, shape=[3, None])
y_data = tf.placeholder(tf.float32, shape=[1, None])

# w_1 = tf.Variable(tf.random_normal([1]), name='weight')
# w_2 = tf.Variable(tf.random_normal([1]), name='weight')
# w_3 = tf.Variable(tf.random_normal([1]), name='weight')
# w = tf.Variable(tf.random_normal([1, 3]), name='bias')
# b = tf.Variable(tf.random_normal([1]), name='bias')
w = tf.Variable([[1.0833952, 0.534335, 0.41554168]], name='bias')
b = tf.Variable([-1.6826605], name='bias')

# hypothesis = w_1 * x_1 + w_2 * x_2 + w_3 * x_3 + b
hypothesis = tf.matmul(w, x_data) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

data = {x_data: [[73, 93, 89, 96, 73], [80, 88, 91, 98, 66], [75, 93, 90, 100, 70]], y_data: [[152, 185, 180, 196, 142]]}

for step in range(100000):
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict=data)
    if (step + 1) % 1000 == 0:
        print(step + 1, cost_val, w_val, b_val)
