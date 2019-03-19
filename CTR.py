import os
import shutil
import numpy as np
import tensorflow as tf
from load import *


class CTR(LoadData):

    def metrics_function(self, function_name):
        if function_name == 'mse':
            return tf.reduce_mean(tf.square(self.y - self.y_predict))
        elif function_name == 'rmse':
            return tf.sqrt(tf.reduce_mean(tf.square(self.y - self.y_predict)))
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

    def FM(self, k=10, l2=0., learning_rate=0.01, loss='mse', optimizer='gd', early_stopping=False):
        # assert
        assert type(k) is int, 'The type of k must be int'
        assert 0. <= l2 <= 1., 'l2 must be in [0, 1]'
        assert type(early_stopping) is bool, 'early_stopping in [False, True]'

        # Input Parameter
        self.k = k
        self.l2 = l2
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping

        # Input
        self.idx = tf.placeholder('float', [None, self.usecol, None])  # [len, field, max_count]
        self.val = tf.placeholder('float', [None, self.usecol, None, 1])  # [len, field, max_count, 1]
        self.y = tf.placeholder('float', [None, 1]) # [len ,1]

        # Weight
        self.w0 = tf.Variable(tf.zeros([1]))
        self.w1 = tf.Variable(tf.zeros([self.num_feature, 1]))   # [feature, 1]
        self.v = tf.Variable(tf.random.normal([self.num_feature, self.k], mean=0, stddev=0.01)) # [feature, k]

        # Linear
        embedding_w1 = tf.nn.embedding_lookup(self.w1, self.idx)   # [len, field, max_count, 1]
        self.linear = tf.multiply(embedding_w1, self.val)

        # pair
        embedding_v = tf.nn.embedding_lookup(self.v, self.idx)  # [len, field, max_count, k]
        pow_multiply = tf.reduce_sum(tf.multiply(tf.pow(embedding_v, 2), tf.pow(self.val, 2)), [1,2])  # [len ,k]
        multiply_pow = tf.reduce_sum(tf.multiply(embedding_v, self.val))
        self.pair = 0.5 * tf.reduce_sum(tf.subtract(tf.pow(, 2), ), axis=1, keep_dims=True) # [len, 1]

        self.pair = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(tf.matmul(self.x,tf.transpose(self.v)),2),
                tf.matmul(tf.pow(self.x,2),tf.transpose(tf.pow(self.v,2)))
            ), axis=1, keep_dims=True)

        self.y_predict = tf.add(self.linear, self.pair)

        self.metrics = self.metrics_function(self.loss)

        lambda_w1 = tf.constant(self.l2, name='lambda_w1')
        lambda_v = tf.constant(self.l2, name='lambda_v')

        # l2
        l2_regularation = tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w1, self.w1),
                tf.multiply(lambda_v, self.v)
            )
        )

        # loss & optimizer
        final_loss = tf.add(self.loss, l2_regularation)
        self.optimizer_function(final_loss)

    def session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def cal_score(self, x, y):
        score = 0
        score += self.sess.run(self.metrics, feed_dict={self.x: x,
                                                        self.y: y})
        return score


    def print_score(self, epoch):
        print(f'After {epoch} epoch, '
              f'Training score is {round(self.trainscore / self.train_num, 2)}, '
              f'valid score is {round(self.validscore / self.valid_num, 2)}')

    def train(self):
        self.sess = self.session()
        self.sess.run(tf.global_variables_initializer())

        if 1 - os.path.exists('.checkpoint'):
            os.mkdir('.checkpoint')

        print('Beginning...')

        for idx, val, y in self.data_generator('train'): # idx, val: [len, field, max_count]
            feed_dict = {
                self.idx: idx,
                self.val: val[:, :, :, np.newaxis],
                self.y: y
            }
            self.sess.run(self.train_opt, feed_dict=feed_dict)

        shutil.rmtree('.checkpoint')

    def predict(self):
        y_predict = []
        for x in self.test_generator:
            y_predict.append(self.sess.run(self.y_predict, feed_dict={self.x: x})[:, 0])
        return np.array(y_predict)

    def close(self):
        self.sess.close()
