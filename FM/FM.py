import numpy as np
import tensorflow as tf


class FM(object):
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.row = 100
        self.col = 100

    def traindata(self, x, y):
        self.x_train = x
        self.y_train = y[:, np.newaxis]
        self.row, self.col = self.x_train.shape

    def validdata(self, x, y):
        self.x_test = x
        self.y_test = y[:, np.newaxis]

    def predictdata(self, x):
        self.x_predict = x

    def metrics_function(self, function_name):
        if function_name == 'mse':
            metrics = tf.reduce_mean(tf.square(self.y - self.y_predict))
        if function_name == 'rmse':
            metrics = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.y_predict)))
        return metrics

    def model(self, k=10, l2=0., learning_rate=0.01, loss_name='mse'):
        self.k = k
        self.l2 = l2
        self.learning_rate = learning_rate

        self.x = tf.placeholder('float', [None, self.col])
        self.y = tf.placeholder('float', [None, 1])

        self.w0 = tf.Variable(tf.zeros([1]))
        self.w1 = tf.Variable(tf.zeros([self.col]))
        self.v = tf.Variable(tf.random.normal([self.k, self.col], mean=0, stddev=0.01))

        self.linear = tf.reduce_sum(tf.multiply(self.w1, self.x), 1, keepdims=True)
        self.pair = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(tf.matmul(self.x,tf.transpose(self.v)),2),
                tf.matmul(tf.pow(self.x,2),tf.transpose(tf.pow(self.v,2)))
            ),axis = 1 , keep_dims=True)
        self.y_predict = tf.add(self.linear, self.pair)

        self.metrics = self.metrics_function(loss_name)

        lambda_w1 = tf.constant(self.l2, name='lambda_w1')
        lambda_v = tf.constant(self.l2, name='lambda_v')

        l2_regularation = tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w1, self.w1),
                tf.multiply(lambda_v, self.v)
            )
        )

        loss = tf.add(self.metrics, l2_regularation)
        self.train_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

    def session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def cal_score(self, x, y):
        score = 0
        num = x.shape[0]
        steps = int(np.ceil(num / self.batch_size))
        for i in range(steps):
            start = (i * self.batch_size) % num
            end = min(start + self.batch_size, num)
            size = end - start
            score += self.sess.run(self.metrics, feed_dict={self.x: x[start: end],
                                                        self.y: y[start: end]}) * size
        score = score / num
        return score

    def train(self, epochs=10, batch_size=1000):
        self.epochs = epochs
        self.batch_size = batch_size
        STEPS = int(np.ceil(self.row / self.batch_size)*self.epochs)

        self.sess = self.session()
        self.sess.run(tf.global_variables_initializer())

        i = 0; num_epoch =1; end=0; print('Beginning...')
        for i in range(STEPS):
            start = (i * self.batch_size) % self.row
            if end == self.row:
                start = 0
            end = min(start+self.batch_size, self.row)
            self.sess.run(self.train_opt,feed_dict={self.x: self.x_train[start: end],
                                                    self.y: self.y_train[start: end]})
            if end == self.batch_size:
                trainscore = self.cal_score(self.x_train, self.y_train)
                validscore = self.cal_score(self.x_test, self.y_test)
                print(f'After {num_epoch} epoch, '
                      f'Training score is {round(trainscore, 2)}, '
                      f'valid score is {round(validscore, 2)}')
                num_epoch += 1

    def predict(self):
        return self.sess.run(self.y_predict, feed_dict={self.x: self.x_predict})[:, 0]

    def close(self):
        self.sess.close()
