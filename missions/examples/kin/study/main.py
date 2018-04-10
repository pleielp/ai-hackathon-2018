# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess


# DONOTCHANGE: They are reserved for nsml
def bind_model(sess, config):

    def save(dir_name, *args):
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    def load(dir_name, *args):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        queries = preprocess(raw_data, config.strmaxlen, hidden_layer_size)
        pred = sess.run(hypothesis, feed_dict={x: queries})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10001)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=100)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    # setting DATASET_PATH
    if not HAS_DATASET and not IS_ON_NSML:
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    output_size = 1
    hidden_layer_size = 200
    learning_rate = 0.0001

    # placeholders and variables
    x = tf.placeholder(tf.float32, [None, hidden_layer_size])
    y = tf.placeholder(tf.float32, [None, output_size])
    w = tf.Variable(tf.random_normal([hidden_layer_size, output_size]), name="w")
    b = tf.Variable(tf.random_normal([output_size]), name="b")

    # computed values
    hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
    cost = -(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

    # start session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # declare saver
    saver = tf.train.Saver()
    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    # train mode
    if config.mode == 'train':

        # load dataset
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen, hidden_layer_size)
        # divide by config.batch
        dataset_len = len(dataset)
        one_batch_size = dataset_len // config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        # train per epoch.
        for epoch in range(config.epochs):

            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):

                _, c = sess.run([train, cost], feed_dict={x: data, y: labels})
                w_v, b_v, h, c, p, a = sess.run([w, b, hypothesis, cost, prediction, accuracy], feed_dict={x: data, y: labels})
            print('epoch:', epoch, "accuracy:", a)
            nsml.report(summary=True, scope=locals(), epoch=epoch, accuracy=accuracy, step=epoch)

            # save session
            if IS_ON_NSML:  # on nsml
                nsml.save(epoch)
            else:  # on local
                if epoch % 1000 == 0:
                    print(b_v)
                    save_path = saver.save(sess, os.path.join('save2', str(epoch) + '.ckpt'))
                    print("Model saved in file: %s" % save_path)

    # local test debug mode
    elif config.mode == 'test_debug':
        # saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('save2')
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            print(checkpoint)
            saver.restore(sess, os.path.join('save2', checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

        # load dataset
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = preprocess(f.readlines(), 100, 200)
        with open(os.path.join(DATASET_PATH, 'train/train_label')) as f:
            labels = np.array([[np.float32(x)] for x in f.readlines()])

        # def answerToqueries(strmaxlen, hidden_layer_size):
        #     # pred = sess.run(hypothesis, feed_dict={x: queries})
        #     # clipped = np.array(pred > config.threshold, dtype=np.int)
        #     h, p, a = sess.run([hypothesis, prediction, accuracy], feed_dict={x: queries, y: labels})
        #     w_v, b_v = sess.run([w, b])
        #     print(list(zip(h.flatten(), p.flatten())))
        #     # print(h, p, a)
        #     print(a, b_v)

        # answerToqueries(config.strmaxlen, hidden_layer_size)

        h, p, a = sess.run([hypothesis, prediction, accuracy], feed_dict={x: queries, y: labels})
        w_v, b_v = sess.run([w, b])
        print(list(zip(h.flatten(), p.flatten())))
        print(a, b_v)

    # local test mode
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)
