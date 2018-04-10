# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
# import matplotlib.pyplot as plt
# import math

from test_char_parser import vectorize_str


class KinQueryDataset:

    def __init__(self, dataset_path: str, max_length: int, hidden_layer_size: int):

        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries = preprocess(f.readlines(), max_length, hidden_layer_size)
        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])
        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        # with open(queries_path, 'rt', encoding='utf8') as f:
        #     self.queries2 = preprocess2(f.readlines(), max_length)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # return self.queries[idx], self.labels[idx]
        return self.queries[idx], self.labels[idx]
        # return self.preprocess2(idx), self.labels[idx]


def preprocess(data: list, max_length: int, hidden_layer_size: int):
    # with open('/home/taeyoung/project/NSML/ai-hackathon-2018/missions/examples/kin/sample_data/kin/train/train_data', 'rt', encoding='utf8') as f: data = f.readlines()
    # '가나', 4 => array([[ 0, 19,  0,  0], [ 2, 19,  0,  0]], dtype=int32)
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    def padding(i, q):
        str_num = len(vectorized_data[i][q])
        if str_num < max_length:
            return vectorized_data[i][q] + [[0] * 6] * (max_length - str_num), str_num
        else:
            return vectorized_data[i][q][:max_length], str_num

    def splitBytab(datum, q_num):
        datum = datum.strip('\n')
        if datum.find('\t') + 1:
            return datum.split('\t')[q_num]
        elif datum.find('\\t') + 1:
            return datum.split('\\t')[q_num]
    vectorized_data = [[vectorize_str(splitBytab(datum, 0)), vectorize_str(splitBytab(datum, 1))] for datum in data]
    x = np.zeros((len(data), hidden_layer_size), dtype=np.float32)
    for i in range(0, len(data)):
        zero_padding = np.zeros((2, max_length, 6), dtype=np.float32)
        zero_padding[0, :], num1 = padding(i, 0)
        zero_padding[1, :], num2 = padding(i, 1)
        tf.reset_default_graph()
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_layer_size, state_is_tuple=True)
        initial_state = cell.zero_state(2, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, zero_padding, initial_state=initial_state, dtype=tf.float32, sequence_length=[num1, num2])
        x_data = (outputs[1][num2 - 1] - outputs[0][num1 - 1])**2
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x[i] = sess.run(x_data)
        print("data loaded", i + 1)
    return x


# def preprocess(data: list, max_length: int):
#     # '가나', 4 => array([[ 0, 19,  0,  0], [ 2, 19,  0,  0]], dtype=int32)
#     """
#      입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
#      기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
#      문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

#     :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
#     :param max_length: 문자열의 최대 길이
#     :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
#     """
#     vectorized_data = [[vectorize_str(datum.split('\t')[0]), vectorize_str(datum.split('\t')[1][:-1])] for datum in data]
#     zero_padding = np.zeros((len(data), 2, max_length, 6), dtype=np.float32)
#     for idx, seq in enumerate(vectorized_data):
#         length = len(seq[0])
#         # print(np.array(seq[0]))
#         # print(zero_padding[idx, 0, :length])
#         # print(length)
#         if length >= max_length:
#             length = max_length
#             zero_padding[idx, 0, :length] = np.array(seq[0])[:length]
#         else:
#             zero_padding[idx, 0, :length] = np.array(seq[0])
#     for idx, seq in enumerate(vectorized_data):
#         length = len(seq[1])
#         # print(length)
#         if length >= max_length:
#             length = max_length
#             zero_padding[idx, 1, :length] = np.array(seq[1])[:length]
#         else:
#             zero_padding[idx, 1, :length] = np.array(seq[1])
#     # zero_padding2 = np.zeros((len(data), 10), dtype=np.float32)
#     # cell = tf.contrib.rnn.BasicLSTMCell(num_units=10, state_is_tuple=True)
#     # initial_state = cell.zero_state(2, tf.float32)
#     # for i in range(len(zero_padding)):
#     #     num1 = len(zero_padding[i][0])
#     #     num2 = len(zero_padding[i][1])
#     #     outputs, _states = tf.nn.dynamic_rnn(cell, np.array(zero_padding[i]), initial_state=initial_state, dtype=tf.float32, sequence_length=[num1, num2])
#     #     with tf.Session() as sess:
#     #         sess.run(tf.global_variables_initializer())
#     #         x_data = (outputs[1][num2 - 1] - outputs[0][num1 - 1])**2
#     #         # print(sess.run(x_data))
#     #         zero_padding2[i, :] = sess.run(x_data)
#     #     print("Data loading", i)
#     # # print(zero_padding2)
#     return zero_padding


# def preprocess2(self, idx):
#     tf.reset_default_graph()
#     zero_padding2 = np.zeros((100, 200), dtype=np.float32)
#     cell = tf.contrib.rnn.BasicLSTMCell(num_units=200, state_is_tuple=True)
#     initial_state = cell.zero_state(2, tf.float32)
#     # for i in range(len(self.queries)):
#     for i, queries in enumerate(self.queries[idx]):
#         num1 = len(queries[0])
#         num2 = len(queries[1])
#         outputs, _states = tf.nn.dynamic_rnn(cell, np.array(queries), initial_state=initial_state, dtype=tf.float32, sequence_length=[num1, num2])
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             x_data = (outputs[1][num2 - 1] - outputs[0][num1 - 1])**2
#             # print(sess.run(x_data))
#             zero_padding2[i, :] = sess.run(x_data)
#         print("Data loading", i)
#     # print(zero_padding2)
#     return zero_padding2


# def preprocess2(data: list, max_length: int):
#     pass
#     # '가나', 4 => array([[ 0, 19,  0,  0], [ 2, 19,  0,  0]], dtype=int32)
#     """
#      입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
#      기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
#      문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

#     :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
#     :param max_length: 문자열의 최대 길이
#     :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
#     """
#     vectorized_data = [[vectorize_str(datum.split('\t')[0]), vectorize_str(datum.split('\t')[1][:-1])] for datum in data]
#     zero_padding = np.zeros((len(data), 2, max_length, 6), dtype=np.float32)
#     for idx, seq in enumerate(vectorized_data):
#         length = len(seq[0])
#         # print(np.array(seq[0]))
#         # print(zero_padding[idx, 0, :length])
#         # print(length)
#         if length >= max_length:
#             length = max_length
#             zero_padding[idx, 0, :length] = np.array(seq[0])[:length]
#         else:
#             zero_padding[idx, 0, :length] = np.array(seq[0])
#     for idx, seq in enumerate(vectorized_data):
#         length = len(seq[1])
#         # print(length)
#         if length >= max_length:
#             length = max_length
#             zero_padding[idx, 1, :length] = np.array(seq[1])[:length]
#         else:
#             zero_padding[idx, 1, :length] = np.array(seq[1])
#     return np.array(vectorized_data)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    dataset = KinQueryDataset('../sample_data/kin/', 100)

    # print(dataset.queries[24])
    # print(dataset.labels)
    # print(np.array(dataset.queries2[108][0]).shape)

    x = tf.placeholder(tf.float32, [None, 10])
    y = tf.placeholder(tf.float32, [None, 1])
    w = tf.Variable(tf.random_normal([10, 1]))
    # b = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([109, 1]))
    hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
    cost = -(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
    train = tf.train.AdamOptimizer(0.001).minimize(cost)

    prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

    # plot_x, plot_y = [], []

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    # for i in range(len(dataset.queries2)):
    # for i in range(len(dataset.queries)):

    # num1 = len(dataset.queries[i][0])
    # num2 = len(dataset.queries[i][1])

    # print(type(dataset.queries[i]))
    # print(type(np.array(dataset.queries2[i])))
    # print(len(dataset.queries2[i][0]))
    # print(len(dataset.queries2[i][1]))
    # print(dataset.queries[i])
    # print(dataset.queries2)
    # sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs[0], targets=outputs[1], weights=tf.constant(1, shape=[10]))

    # initial_state = cell.zero_state(2, tf.float32)
    # outputs, _states = tf.nn.dynamic_rnn(cell, np.array(dataset.queries[i]), initial_state=initial_state, dtype=tf.float32, sequence_length=[num1, num2])

    # num3 = len(dataset.queries2[38][0])
    # num4 = len(dataset.queries2[38][1])

    # outputs, _states = tf.nn.dynamic_rnn(cell, dataset.queries[38], initial_state=initial_state, dtype=tf.float32, sequence_length=[num3, num4])

    # print(i)
    # print(sess.run(outputs))
    # print(sess.run(outputs)[0][num1 - 1])
    # print(sess.run(outputs)[1][num2 - 1])
    # tf.reduce_mean((outputs[1][num2 - 1] - outputs[0][num2 - 1])**2)

    # with open('test.txt', 'a') as f:

    # f.write(str(sess.run(tf.as_string(tf.reduce_mean((outputs[1][num2 - 1] - outputs[0][num1 - 1])**2)))) + "\n")

    # print(sess.run(outputs).shape)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     x_data = (outputs[1][num2 - 1] - outputs[0][num1 - 1])**2
    #     print(sess.run(x_data))
    #     plot_y.append(sess.run(x_data))

    # print(sess.run(tf.reduce_mean((outputs[1][num2 - 1] - outputs[0][num1 - 1])**2)), dataset.labels[i])
    # result = sess.run(tf.reduce_mean((outputs[1][num2 - 1] - outputs[0][num1 - 1])**2))
    # print(type(result.item()))
    # print(math.log(result.item()) + 27)
    # print(sess.run(tf.sigmoid(np.array([math.log(result.item()) + 27]))))
    # hypothesis = tf.sigmoid(np.array([math.log(result.item()) + 27]))

    # print(sess.run(sequence_loss))
    # print(sess.run(outputs2)[0][num3 - 1], sess.run(outputs2)[1][num4 - 1])
    # print(sess.run(outputs)[0][num1 - 1] - sess.run(outputs2)[1][num4 - 1])

    # for i in range(len(data)): data2.append(float(data[i][2:-2])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    #     print(np.array(plot_y))
    #     # plot_x = [1] * len(dataset.queries)
    #     # plt.scatter(plot_x, plot_y)
    #     # plt.show()

    #     # y = dataset.labels[i]
        for step in range(2):
            cost_train, _ = sess.run([cost, train], feed_dict={x: dataset.queries, y: dataset.labels})
            print(step, "  cost: ", cost_train)
    #     # print(sess.run(hypothesis), feed_dict={x: plot_y, y: dataset.labels})
    #     # print(sess.run(hypothesis), y, sess.run(cost))

        h, p, a = sess.run([hypothesis, prediction, accuracy], feed_dict={x: dataset.queries, y: dataset.labels})
        print("\nhypothesis: ", h, " prediction: ", p, " accuracy: ", a)
