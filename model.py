import tensorflow as tf
from load import *


class Model(LoadData):

    def linear_regression(self, learning_rate=0.1, loss='mse', optimizer='gd'):
        # Input Parameter
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer

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
            embedding_w1 = tf.nn.embedding_lookup(self.w1, self.tensor_idx)   # [len, field, max_count, 1]
            self.linear = tf.reduce_sum(tf.multiply(embedding_w1, self.tensor_val), [1, 2])  # [len, 1]
            self.y_predict = tf.add(self.linear, self.w0)  # [len, 1]

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

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Input
            self.tensor_idx = tf.placeholder(tf.int32, [None, len(self.usecol), None])  # [len, field, max_count]
            self.tensor_val = tf.placeholder(tf.float32, [None, len(self.usecol), None, 1])  # [len, field, max_count, 1]
            self.tensor_y = tf.placeholder(tf.float32, [None, 1])  # [len ,1]

            # Weight
            self.w0 = tf.Variable(tf.zeros([1]))
            self.w1 = tf.Variable(tf.zeros([self.num_feature, 1]))   # [feature, 1]
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
            self.metrics = self.metrics_function(self.loss)

            # l2
            lambda_w1 = tf.constant(self.l2, name='lambda_w1')
            lambda_v = tf.constant(self.l2, name='lambda_v')

            l2_regularation = tf.reduce_sum(
                tf.add(
                    tf.multiply(lambda_w1, tf.pow(self.w1, 2)),
                    tf.multiply(lambda_v, tf.pow(self.v, 2))
                )
            )

            # loss & optimizer
            final_loss = tf.add(self.metrics, l2_regularation)
            self.optimizer_function(final_loss)

            self.saver = tf.train.Saver()

    def DeepFM(self, ):
