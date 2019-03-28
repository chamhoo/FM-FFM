import os
import shutil
import numpy as np
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from model import *
from plot import *


class CTR(Model):

    def _session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(graph=self.graph, config=config)

    def _blank_score(self):
        self.train_score = 0
        self.valid_score = 0
        self.train_len = 0
        self.valid_len = 0

    def _checkpoint(self, num_epoch):
        self.saver.save(self.sess, f'.checkpoint/epoch{num_epoch}/model.ckpt')

    def train(self, epoch, early_stopping, verbose, early_stopping_epoch=1, retrain=False):
        """
        :param epoch: int
        :param early_stopping: [True, False]
        :param verbose: Print information level.
               0=silent, 1=just finally score, 2=progress bar.
        :param early_stopping_epoch: will stop training if the metric of valid data
               doesn't improve in last {early_stopping_epoch} epoch, This parameter
               works only when early_stopping is True.
        :param retrain: How do you want to train your data?
               From the beginning or load an existing parameter for the model?
               Adjust this switch to True if you wish to load existing parameters,
               otherwise False.
               WARNING: As well as an important thing, please see the note for load_model()
        """
        # assert
        assert type(epoch) is int, 'The type of epoch is int'
        assert type(early_stopping) is bool, 'early_stopping in [False, True]'
        assert verbose in [0, 1, 2], 'verbose is out of range [0, 1, 2]'
        assert type(early_stopping_epoch) is int, 'The type of early_stopping_epoch is int'
        assert type(retrain) is bool, 'The type of retrain is bool'

        best_epoch = 0
        start_epoch = 1
        error_rise_count = 0
        recorder = {'train_score': [], 'valid_score': []}

        # built sess
        with self._session() as self.sess:
            if retrain is True:
                self.saver.restore(self.sess, self.modelinfo['ckpt'])
                start_epoch = self.modelinfo['start_epoch']
                recorder = self.modelinfo['recorder']
            else:
                self.sess.run(tf.global_variables_initializer())

            if 1 - os.path.exists('.checkpoint'):
                os.mkdir('.checkpoint')

            for num_epoch in range(start_epoch, start_epoch+epoch):
                self._blank_score()
                for idx, val, y, batch_size in self.data_generator('train'):  # idx, val: [len, field, max_count]
                    self.train_len += batch_size
                    feed_dict = {self.tensor_idx: idx,
                                 self.tensor_val: val[:, :, :, np.newaxis],
                                 self.tensor_y: y}
                    self.sess.run(self.train_opt, feed_dict=feed_dict)
                    self.train_score += self.sess.run(self.metrics, feed_dict=feed_dict) * batch_size
                self.train_score = self.train_score / self.train_len

                for idx, val, y, batch_size in self.data_generator('valid'):
                    self.valid_len += batch_size
                    feed_dict = {self.tensor_idx: idx,
                                 self.tensor_val: val[:, :, :, np.newaxis],
                                 self.tensor_y: y}
                    self.valid_score += self.sess.run(self.metrics, feed_dict=feed_dict) * batch_size
                self.valid_score = self.valid_score / self.valid_len

                recorder['train_score'].append(self.train_score)
                recorder['valid_score'].append(self.valid_score)

                # print score
                if verbose == 2:
                    print(f'After {num_epoch} epoch,'
                          f' train score is {round(self.train_score, 4)}, '
                          f'valid score is {round(self.valid_score, 4)}')

                # save & early stopping
                self._checkpoint(num_epoch)
                if early_stopping:
                    best_epoch = recorder['valid_score'].index(min(recorder['valid_score'])) + 1
                    if best_epoch != num_epoch:
                        error_rise_count += 1
                        if error_rise_count >= early_stopping_epoch:
                            break
                else:
                    best_epoch = num_epoch

            # print best score
            if verbose in [1, 2]:
                print(f'best epoch is {best_epoch}, '
                      f' train score is {recorder["train_score"][best_epoch-1]}, '
                      f'valid score is {recorder["valid_score"][best_epoch-1]}')

            # update modelinfo
            self.modelinfo = {'start_epoch': best_epoch + 1,
                              'ckpt': f'.checkpoint/epoch{best_epoch}/model.ckpt',
                              'recorder': recorder}

    def load_model(self, ckpt, start_epoch=1, recorder=None):
        """
        If you wish to continue use an existing model,
        this function is used to enter the start
        epoch and the location of the ckpt file.
        If you have just run the train function,
        then you don't need to run this function
        because the train function has already recorded the above information.
        But if not, you need to run the function to set this infomation.
        """
        if recorder is None:
            recorder = {'train_score': [2**32], 'valid_score': [2**32]}
        self.modelinfo = {'start_epoch': start_epoch, 'ckpt': ckpt, 'recorder': recorder}

    def cv(self,
           epoch,
           early_stopping=True,
           verbose=False,
           nfolds=5,
           early_stopping_epoch=1):
        """
        Perform the cross-validation.
        :param epoch: Defined in train function, type is int.
        :param early_stopping: Defined in train function(default=True).
        :param verbose: bool, optional(default=False), False=silent, True=print
        :param nfolds: int, default=5, Number of folds in CV.
        :param early_stopping_epoch: Defined in train function, type is int.
        :return: a dict, key values in ['mean_score', 'mean_std', 'history'],
                 mean_score: mean score of {nfolds} train.
                 std: standard deviation of {nflods} train.
                 history: a dict has the following format:
                 {
                    'flod_1':{'train_score': a list of train_score, len is the actual number of epoch,
                              'valid_score': a list of valid_score, len is the actual number of epoch},
                    'flod2': {'train_score': [...],
                              'valid_score': [...]},
                    ...
                    'flodn': {'train_score': [...],
                              'valid_score': [...]},
                }
        """

        # assert
        assert type(verbose) is bool, 'The type of verbose is bool'
        assert type(nfolds) is int, 'The type of nflods is int'

        # train parameter
        train_param = dict()
        train_param['verbose'] = 2
        train_param['epoch'] = epoch
        train_param['retrain'] = False
        train_param['early_stopping'] = early_stopping
        train_param['early_stopping_epoch'] = early_stopping_epoch

        # result
        score_list = []
        result_dict = dict()
        result_dict['history'] = dict()

        # random set seed & training
        seed_gen = self._pseudo_random(1664525, 1013904223, False)
        for fold in range(1, 1+nfolds):
            self.reset_seed(next(seed_gen))
            self.train(**train_param)
            best_epoch = self.modelinfo['start_epoch'] - 1
            best_score = self.modelinfo['recorder']['valid_score'][best_epoch - 1]
            score_list.append(best_score)
            # shutil.rmtree('.checkpoint')
            result_dict['history'][f'fold{fold}'] = self.modelinfo['recorder']

            if verbose is True:
                print(f'Flod {fold} is done,'
                      f'best score is {round(best_score, 4)}')

        result_dict['mean_score'] = np.mean(score_list)
        result_dict['mean_std'] = np.std(score_list)
        if verbose is True:
            print(f'Finally,'
                  f' mean score is {result_dict["mean_score"]}, '
                  f'std is {result_dict["mean_std"]}')

        return result_dict

    def predict(self):
        y_predict = np.array([])
        with self._session() as sess:
            for idx, val, batch_size in self.data_generator('test'):
                feed_dict = {self.tensor_idx: idx,
                             self.tensor_val: val[:, :, :, np.newaxis]}
                np.append(y_predict, sess.run(self.y_predict, feed_dict=feed_dict))
        return y_predict
