import os
import shutil
import numpy as np
import tensorflow as tf
from load import *


class CTR(LoadData):

    def metrics_function(self, function_name):
        """
        Lower is better
        """
        if function_name == 'mse':
            return tf.reduce_mean(tf.square(self.tensor_y - self.y_predict))
        elif function_name == 'rmse':
            return tf.sqrt(tf.reduce_mean(tf.square(self.tensor_y - self.y_predict)))
        else:
            assert False, 'metrics name is not exit'

    def optimizer_function(self, loss):
        if self.optimizer == 'adam':
            self.train_opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            ).minimize(loss)

        elif self.optimizer == 'adagrad':
            self.train_opt = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate,
                initial_accumulator_value=1e-8
            ).minimize(loss)

        elif self.optimizer == 'gd':
            self.train_opt = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate
            ).minimize(loss)

        elif self.optimizer == 'momentun':
            self.train_opt = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=0.95
            ).minimize(loss)

        else:
            assert False, 'optimizer name is not exit'

    def linear_regression(self, learning_rate=0.01, loss='mse', optimizer='gd'):
        # Input Parameter
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer

        # Input
        self.tensor_idx = tf.placeholder(tf.int32, [None, len(self.usecol), None])  # [len, field, max_count]
        self.tensor_val = tf.placeholder(tf.float32, [None, len(self.usecol), None, 1])  # [len, field, max_count, 1]
        self.tensor_y = tf.placeholder(tf.float32, [None, 1])  # [len ,1]

        # Weight
        self.w0 = tf.Variable(tf.ones([1]))
        self.w1 = tf.Variable(tf.ones([self.num_feature, 1]))

        # Linear
        embedding_w1 = tf.nn.embedding_lookup(self.w1, self.tensor_idx)   # [len, field, max_count, 1]
        self.linear = tf.reduce_sum(tf.multiply(embedding_w1, self.tensor_val), [1, 2])  # [len, 1]
        self.y_predict = tf.add(self.linear, self.w0)

        # metrics
        self.metrics = self.metrics_function(self.loss)
        self.optimizer_function(self.metrics)

        self.saver = tf.train.Saver()


    def FM(self, k=10, l2=0., learning_rate=0.01, loss='mse', optimizer='gd'):
        # assert
        assert type(k) is int, 'The type of k must be int'
        assert 0. <= l2 <= 1., 'l2 must be in [0, 1]'

        # Input Parameter
        self.k = k
        self.l2 = l2
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # Input
        self.tensor_idx = tf.placeholder(tf.int32, [None, len(self.usecol), None])  # [len, field, max_count]
        self.tensor_val = tf.placeholder(tf.float32, [None, len(self.usecol), None, 1])  # [len, field, max_count, 1]
        self.tensor_y = tf.placeholder(tf.float32, [None, 1])  # [len ,1]

        # Weight
        self.w0 = tf.Variable(tf.zeros([1]))
        self.w1 = tf.Variable(tf.zeros([self.num_feature, 1]))   # [feature, 1]
        self.v = tf.Variable(tf.random.normal([self.num_feature, self.k], mean=0, stddev=0.01)) # [feature, k]

        # Linear
        embedding_w1 = tf.nn.embedding_lookup(self.w1, self.tensor_idx)   # [len, field, max_count, 1]
        self.linear = tf.reduce_sum(tf.multiply(embedding_w1, self.tensor_val), [1, 2])  # [len, 1]

        # pair
        embedding_v = tf.nn.embedding_lookup(self.v, self.tensor_idx)  # [len, field, max_count, k]
        pow_multiply = tf.reduce_sum(tf.multiply(tf.pow(embedding_v, 2), tf.pow(self.tensor_val, 2)), [1,2])  # [len ,k]
        multiply_pow = tf.pow(tf.reduce_sum(tf.multiply(embedding_v, self.tensor_val), [1, 2]), 2)  # [len, k]
        self.pair = 0.5 * tf.reduce_sum(tf.subtract(multiply_pow, pow_multiply), axis=1, keep_dims=True)  # [len, 1]

        self.y_predict = tf.add(self.linear, self.pair)
        self.metrics = self.metrics_function(self.loss)

        # l2
        lambda_w1 = tf.constant(self.l2, name='lambda_w1')
        lambda_v = tf.constant(self.l2, name='lambda_v')

        l2_regularation = tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w1, self.w1),
                tf.multiply(lambda_v, self.v)
            )
        )

        # loss & optimizer
        final_loss = tf.add(self.metrics, l2_regularation)
        self.optimizer_function(final_loss)

        self.saver = tf.train.Saver()

    def _session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def blank_score(self):
        self.train_score = 0
        self.valid_score = 0
        self.train_len = 0
        self.valid_len = 0

    def _checkpoint(self, num_epoch):
        self.saver.save(self.sess, f'.checkpoint/epoch{num_epoch}/model.ckpt')
        useless_ckpt = f'.checkpoint/epoch{num_epoch - self.early_stopping_epoch}'
        if os.path.exists(useless_ckpt):
            shutil.rmtree(useless_ckpt)

    def train(self, epoch, early_stopping, verbose, early_stopping_epoch=1, save_recorder=False):
        assert type(epoch) is int, 'The type of epoch is int'
        assert type(early_stopping) is bool, 'early_stopping in [False, True]'
        assert type(verbose) is bool, 'verbose: False=silent, True=progress bar '
        assert type(early_stopping_epoch) is int, 'The type of early_stopping_epoch is int'
        assert type(save_recorder) is bool, ''

        last_score = 2**32    # 2**32: No specific meaning, just a large enough number.
        self.epoch = epoch
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.early_stopping_epoch = early_stopping_epoch
        self.blank_score()
        self.recorder = {'train_score': [], 'valid_score': []}

        self.sess = self._session()
        self.sess.run(tf.global_variables_initializer())

        if 1 - os.path.exists('.checkpoint'):
            os.mkdir('.checkpoint')

        print('Beginning...')

        for num_epoch in range(self.epoch):
            for idx, val, y, batch_size in self.data_generator('train'):  # idx, val: [len, field, max_count]

                self.train_len += batch_size
                feed_dict = {
                    self.tensor_idx: idx,
                    self.tensor_val: val[:, :, :, np.newaxis],
                    self.tensor_y: y
                }
                self.sess.run(self.train_opt, feed_dict=feed_dict)
                self.train_score += self.sess.run(self.metrics, feed_dict=feed_dict) * batch_size
            self.train_score = self.train_score / self.train_len

            for idx, val, y, batch_size in self.data_generator('valid'):
                self.valid_len += batch_size
                feed_dict = {
                    self.tensor_idx: idx,
                    self.tensor_val: val[:, :, :, np.newaxis],
                    self.tensor_y: y
                }
                self.valid_score += self.sess.run(self.metrics, feed_dict=feed_dict) * batch_size
            self.valid_score = self.valid_score / self.valid_len
            """
            self.recorder['train_score'].append(self.train_score)
            self.recorder['valid_score'].append(self.valid_score)

            # print score
            if self.verbose:
                print(f'After {num_epoch} epoch,'
                      f' train score is {self.train_score}, '
                      f'valid score is {self.valid_score}')

            # save & early stopping

            if early_stopping:
                if last_score > self.valid_score:
                    last_score = self.valid_score
                    self._checkpoint(num_epoch)
                else:
                    break
            else:
                self._checkpoint(num_epoch)
            """

        shutil.rmtree('.checkpoint')

    def predict(self):
        y_predict = []
        for x in self.test_generator:
            y_predict.append(self.sess.run(self.y_predict, feed_dict={self.x: x})[:, 0])
        return np.array(y_predict)

    def close(self):
        self.sess.close()
