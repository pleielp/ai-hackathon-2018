import tensorflow as tf

#
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv', 'data-02-test-score.csv'], shuffle=False, name='filename_quere')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[:-1], xy[-1:]], batch_size=10)

x_data = tf.placeholder(tf.float32, shape=[3, None])
y_data = tf.placeholder(tf.float32, shape=[1, None])

w = tf.Variable([[1.0833952, 0.534335, 0.41554168]], name='bias')
b = tf.Variable([-1.6826605], name='bias')

hypothesis = tf.matmul(w, x_data) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

data = {x_data: [[73, 93, 89, 96, 73], [80, 88, 91, 98, 66], [75, 93, 90, 100, 70]], y_data: [[152, 185, 180, 196, 142]]}

for step in range(100000):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict=data)
    if (step + 1) % 1000 == 0:
        print(step + 1, cost_val, w_val, b_val)

#
coord.request_stop()
coord.join(threads)
