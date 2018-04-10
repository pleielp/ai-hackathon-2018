# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from test_char_parser import vectorize_str


class KinQueryDataset:

    def __init__(self, dataset_path: str, max_length: int):

        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries = preprocess(f.readlines(), max_length)
        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])
        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries2 = preprocess2(f.readlines(), max_length)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]


def preprocess(data: list, max_length: int):
    # '가나', 4 => array([[ 0, 19,  0,  0], [ 2, 19,  0,  0]], dtype=int32)
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    vectorized_data = [[vectorize_str(datum.split('\t')[0]), vectorize_str(datum.split('\t')[1][:-1])] for datum in data]
    zero_padding = np.zeros((len(data), 2, max_length, 6), dtype=np.float32)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq[0])
        # print(np.array(seq[0]))
        # print(zero_padding[idx, 0, :length])
        # print(length)
        if length >= max_length:
            length = max_length
            zero_padding[idx, 0, :length] = np.array(seq[0])[:length]
        else:
            zero_padding[idx, 0, :length] = np.array(seq[0])
    for idx, seq in enumerate(vectorized_data):
        length = len(seq[1])
        # print(length)
        if length >= max_length:
            length = max_length
            zero_padding[idx, 1, :length] = np.array(seq[1])[:length]
        else:
            zero_padding[idx, 1, :length] = np.array(seq[1])
    return zero_padding


def preprocess2(data: list, max_length: int):
    # '가나', 4 => array([[ 0, 19,  0,  0], [ 2, 19,  0,  0]], dtype=int32)
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    vectorized_data = [[vectorize_str(datum.split('\t')[0]), vectorize_str(datum.split('\t')[1][:-1])] for datum in data]
    # zero_padding = np.zeros((len(data), 2, max_length, 6), dtype=np.float32)
    # for idx, seq in enumerate(vectorized_data):
    #     length = len(seq[0])
    #     # print(np.array(seq[0]))
    #     # print(zero_padding[idx, 0, :length])
    #     # print(length)
    #     if length >= max_length:
    #         length = max_length
    #         zero_padding[idx, 0, :length] = np.array(seq[0])[:length]
    #     else:
    #         zero_padding[idx, 0, :length] = np.array(seq[0])
    # for idx, seq in enumerate(vectorized_data):
    #     length = len(seq[1])
    #     # print(length)
    #     if length >= max_length:
    #         length = max_length
    #         zero_padding[idx, 1, :length] = np.array(seq[1])[:length]
    #     else:
    #         zero_padding[idx, 1, :length] = np.array(seq[1])
    return vectorized_data


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    dataset = KinQueryDataset('../sample_data/kin/', 101)

    # print(dataset.queries[24])
    # print(dataset.labels)
    # print(np.array(dataset.queries2[108][0]).shape)

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=10, state_is_tuple=True)

    Weight = tf.Variable(tf.zeros([109, 20, 1]))
    bias = tf.Variable(tf.zeros([1]))

    2410214202
    # x_train_input = tf.Variable(tf.zeros([20,109]))
    x_train_input = tf.placeholder(tf.float32, shape=[109, 1, 20])

    h = tf.matmul(x_train_input, Weight) + bias
    y_predict = tf.div(1., 1. + tf.exp(-h))

    loss = tf.reduce_mean(tf.square(y_predict - dataset.labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    ttrain = optimizer.minimize(loss)

    result = []

    for i in range(len(dataset.queries2)):
    # for i in range(len(dataset.queries)):
        num1 = len(dataset.queries2[i][0])
        num2 = len(dataset.queries2[i][1])

        initial_state = cell.zero_state(2, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, dataset.queries[i], initial_state=initial_state, dtype=tf.float32, sequence_length=[num1, num2])
        # sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs[0], targets=outputs[1], weights=tf.constant(1, shape=[10]))

        # num3 = len(dataset.queries2[38][0])
        # num4 = len(dataset.queries2[38][1])

        # outputs2, _states = tf.nn.dynamic_rnn(cell, dataset.queries[38], initial_state=initial_state, dtype=tf.float32, sequence_length=[num3, num4])

        print(i, outputs)
        # result.append(outputs)

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
            # print(sess.run(outputs))
            # with open('test2.txt', 'a') as f:
            #     f.write(str(sess.run(tf.as_string(outputs))))



    for step in range(100): 
        with tf.Session() as sess:
            print(tf.concat(outputs,1))
            sess.run(tf.global_variables_initializer())
            sess.run(ttrain, feed_dict={x_train_input: np.array(tf.concat(outputs, 1))})
            print(step, sess.run(loss))

    print(sess.run(y_predict))
    