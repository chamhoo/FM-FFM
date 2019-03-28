import tensorflow as tf
from load import *
from plot import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class Model(LoadData):

    def _metrics_function(self, function_name):
        """
        Lower is better
        """
        if function_name == 'mse':
            return tf.reduce_mean(tf.square(self.tensor_y - self.y_predict))
        elif function_name == 'rmse':
            return tf.sqrt(tf.reduce_mean(tf.square(self.tensor_y - self.y_predict)))
        else:
            assert False, 'metrics name is not exit'

    def _optimizer_function(self, loss):
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

    def linear_regression(self, learning_rate=0.1, l2=0., loss='mse', optimizer='gd'):
        # Input Parameter
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.l2 = l2

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Input
            self.tensor_idx = tf.placeholder(tf.int32, [None, len(self.usecol), None])  # [len, field, max_count]
            self.tensor_val = tf.placeholder(tf.float32, [None, len(self.usecol), None, 1])  # [len, field, max_count, 1]
            self.tensor_y = tf.placeholder(tf.float32, [None, 1])  # [len ,1]

            # Weight
            self.w0 = tf.Variable(tf.ones([1]))
            self.w1 = tf.Variable(tf.ones([self.num_feature, 1]))

            # Linear
            embedding_w1 = tf.nn.embedding_lookup(self.w1, self.tensor_idx, validate_indices=False)   # [len, field, max_count, 1]
            self.linear = tf.reduce_sum(tf.multiply(embedding_w1, self.tensor_val), [1, 2])  # [len, 1]
            self.y_predict = tf.add(self.linear, self.w0)  # [len, 1]

            # metrics
            self.metrics = self._metrics_function(self.loss)

            # l2
            l2_regularation = tf.constant(0, dtype=tf.float32)
            if self.l2 > 0:
                l2_regularation = tf.add(
                    tf.contrib.layers.l2_regularizer(self.l2)(self.w1),
                    tf.contrib.layers.l2_regularizer(self.l2)(self.v)
                )

            # loss & optimizer
            final_loss = tf.add(self.metrics, l2_regularation)
            self._optimizer_function(final_loss)

            self.saver = tf.train.Saver()

    def FM(self, k=10, l2=0., learning_rate=0.01, loss='mse', optimizer='gd'):
        """
        Factorization Machine
        - https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
        :param k: int, (default=10)
        :param l2: L2 regularization
        :param learning_rate: Decide on the speed of training.
               The smaller the value, the more accurate the value will be,
               but at the same time it will take more time.
               Conversely, the bigger the result may not be so good, but the speed is faster.
        :param loss: string, default='mse', list can be found in README.md, the name of loss function.
        :param optimizer: string, default='gd', list can be found in README.md, the name of optimizer function.
        """
        # assert
        assert type(k) is int, 'The type of k must be int'
        assert 0. <= l2 <= 1., 'l2 must be in [0, 1]'

        # Input Parameter
        self.k = k
        self.l2 = l2
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Input
            self.tensor_idx = tf.placeholder(tf.int32, [None, len(self.usecol), None])  # [len, field, max_count]
            self.tensor_val = tf.placeholder(tf.float32, [None, len(self.usecol), None, 1])  # [len, field, max_count, 1]
            self.tensor_y = tf.placeholder(tf.float32, [None, 1])  # [len ,1]

            # Weight
            self.w0 = tf.Variable(tf.ones([1]))
            self.w1 = tf.Variable(tf.ones([self.num_feature, 1]))   # [feature, 1]
            self.v = tf.Variable(tf.random.normal([self.num_feature, self.k], mean=0, stddev=0.01))  # [feature, k]

            # Linear
            embedding_w1 = tf.nn.embedding_lookup(self.w1, self.tensor_idx)   # [len, field, max_count, 1]
            self.linear = tf.reduce_sum(tf.multiply(embedding_w1, self.tensor_val), [1, 2])  # [len, 1]

            # pair
            embedding_v = tf.nn.embedding_lookup(self.v, self.tensor_idx)  # [len, field, max_count, k]
            pow_multiply = tf.reduce_sum(tf.multiply(tf.pow(embedding_v, 2), tf.pow(self.tensor_val, 2)), [1,2])  # [len ,k]
            multiply_pow = tf.pow(tf.reduce_sum(tf.multiply(embedding_v, self.tensor_val), [1, 2]), 2)  # [len, k]
            self.pair = 0.5 * tf.reduce_sum(tf.subtract(multiply_pow, pow_multiply), axis=1, keepdims=True)  # [len, 1]

            self.y_predict = tf.add(self.linear, self.pair)
            self.metrics = self._metrics_function(self.loss)

            # l2
            l2_regularation = tf.constant(0, dtype=tf.float32)
            if self.l2 > 0:
                l2_regularation = tf.add(
                    tf.contrib.layers.l2_regularizer(self.l2)(self.w1),
                    tf.contrib.layers.l2_regularizer(self.l2)(self.v)
                )

            # loss & optimizer
            final_loss = tf.add(self.metrics, l2_regularation)
            self._optimizer_function(final_loss)

            self.saver = tf.train.Saver()

    def FFM(self, k=10, l2=0., learning_rate=0.01, loss='mse', optimizer='gd'):
        """
        Field-aware Factorization Machines
        - https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
        :param k: int, (default=10)
        :param l2: L2 regularization
        :param learning_rate: Decide on the speed of training.
               The smaller the value, the more accurate the value will be,
               but at the same time it will take more time.
               Conversely, the bigger the result may not be so good, but the speed is faster.
        :param loss: string, (default='mse'), list can be found in README.md, the name of loss function.
        :param optimizer: string, (default='gd'), list can be found in README.md, the name of optimizer function.
        """
