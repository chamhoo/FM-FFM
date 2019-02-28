import numpy as np
import tensorflow as tf


class FM(object):
    def __init__(self):
        self.train_generator = None
        self.test_generator = None
        self.col = 100
        self.row = 100

    def info_reseive(self, col):
        self.col = col

    def traindata(self, train):
        self.train_generator = train

    def predictdata(self, test):
        self.test_generator = test

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
        score += self.sess.run(self.metrics, feed_dict={self.x: x,
                                                        self.y: y})
        return score

    def random_split(self, percentage=80):
        while True:
            rand = np.random.randint(0, 100, 1)
            if percentage > rand:
                yield True
            else:
                yield False

    def blank(self):
        self.train_num = 0
        self.valid_num = 0
        self.trainscore = 0
        self.validscore = 0

    def print_score(self, epoch):
        print(f'After {epoch} epoch, '
              f'Training score is {round(self.trainscore / self.train_num, 2)}, '
              f'valid score is {round(self.validscore / self.valid_num, 2)}')

    def train(self):
        rand = self.random_split()

        self.sess = self.session()
        self.sess.run(tf.global_variables_initializer())

        print('Beginning...')

        last_epoch = 0
        self.blank()

        for x, y, epoch in self.train_generator:
            if next(rand):
                self.sess.run(self.train_opt,feed_dict={self.x: x,
                                                        self.y: y})
                sector_len = len(x)
                self.trainscore += self.cal_score(x, y) * sector_len
                self.train_num += sector_len
            else:
                sector_len = len(x)
                self.validscore += self.cal_score(x, y) * sector_len
                self.valid_num += sector_len

            if last_epoch != epoch:
                self.print_score(epoch)
                self.blank()
            last_epoch = epoch

        self.print_score(epoch+1)

    def predict(self):
        y_predict = []
        for x in self.test_generator:
            y_predict.append(self.sess.run(self.y_predict, feed_dict={self.x: x})[:, 0])
        return np.array(y_predict)

    def close(self):
        self.sess.close()
