# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

from char_parser import vectorize_str


class KinQueryDataset:

    def __init__(self, dataset_path: str, max_length: int, hidden_layer_size: int):

        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # load queries and labels
        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries = preprocess(f.readlines(), max_length, hidden_layer_size)
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]


def preprocess(data: list, max_length: int, hidden_layer_size: int):
    # input: (data_length, 1(문자열)) shape list
    # output: (data_length, 1(유사도))) shape np.array

    # '/t' 또는 '//t'로 분리되는 질문 쌍을 두 개의 질문으로 분리
    def splitBytab(datum, q_num):
        datum = datum.strip('\n')
        if datum.find('\t') + 1:
            return datum.split('\t')[q_num]
        elif datum.find('\\t') + 1:
            return datum.split('\\t')[q_num]

    # 각 질문벡터를 max_length로 zero padding
    def padding(i, q):
        str_num = len(vectorized_data[i][q])
        if str_num < max_length:
            return vectorized_data[i][q] + [[0] * 6] * (max_length - str_num), str_num
        else:
            return vectorized_data[i][q][:max_length], str_num

    # 질문 쌍을 두 개의 질문으로 분리 후 벡터화, (data_length, 2, string_length, 6)
    vectorized_data = [[vectorize_str(splitBytab(datum, 0)), vectorize_str(splitBytab(datum, 1))] for datum in data]
    x = np.zeros((len(data), hidden_layer_size), dtype=np.float32)

    for i in range(0, len(data)):
        # zero padding
        zero_padding = np.zeros((2, max_length, 6), dtype=np.float32)
        zero_padding[0, :], num1 = padding(i, 0)
        zero_padding[1, :], num2 = padding(i, 1)

        # rnn 처리
        tf.reset_default_graph()
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_layer_size, state_is_tuple=True)
        initial_state = cell.zero_state(2, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, zero_padding, initial_state=initial_state, dtype=tf.float32, sequence_length=[num1, num2])

        # 두 질문 간 유사도
        x_data = (outputs[1][num2 - 1] - outputs[0][num1 - 1])**2
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x[i] = sess.run(x_data)
        print("data loaded", i + 1)
    return x
