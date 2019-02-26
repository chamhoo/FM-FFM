"""
 In this example training tasks, we use movielens 100k as training data.
 The movielens 100k's description file location is in ../data/ml-100k/README
 Suggested path = '../data/ml-100k/u.data'
 The col name is [user, movie, score, Timestamp]
"""

import numpy as np


class LoadData(object):
    def __init__(self,
                 trainpath,
                 testpath,
                 targetcol, target_type,
                 batch_size=None,
                 cols=[], unusecol=[], discrete_col=[], numerical_col=[]):

        self.batch_size = batch_size
        self.targetcol = targetcol
        self.target_type = target_type
        self.discrete_col = discrete_col
        self.numerical_col = numerical_col
        self.unusecol = unusecol
        self.usecol = self.discrete_col + self.numerical_col
        self.targetdict = dict()
        self.field_start = dict()
        self.target_nunique = 0
        self.feature_len = 0
        self.fieldlen = dict(zip(self.usecol, [0 for i in self.usecol]))
        self.ledict = dict(zip(self.usecol, [dict() for i in self.usecol]))
        self.data_info = {
            'train':{'path':trainpath, 'cols':cols, 'len':0},
            'test':{'path':testpath, 'cols':list(set(cols) - set(targetcol)), 'len':0}
        }
        self.preload()

    def preload(self):
        """
        preload 的作用仅仅是计算出 fieldlen
        """
        for dataset in ['train', 'test']:
            with open(self.data_info[dataset]['path']) as file:
                for line in file.readlines():
                    line = line.strip().split()
                    i = 0
                    for feature in line:
                        field = self.data_info[dataset]['cols'][i]

                        if field in self.numerical_col:
                            self.ledict[field] = 0
                            self.fieldlen[field] = 1

                        if field in self.discrete_col:
                            isexit = self.ledict[field].get(feature, False)
                            if isexit is False:
                                self.ledict[field][feature] = self.fieldlen[field]
                                self.fieldlen[field] += 1

                        if field == self.targetcol:
                            if self.target_type == 'discrete':
                                isexit = self.targetdict.get(feature, False)
                                if isexit is False:
                                    self.targetdict[feature] = self.target_nunique
                                    self.target_nunique += 1

                        if field in self.unusecol:
                            pass

                        i += 1
                    self.data_info[dataset]['len'] += 1

        i = 0
        for key, value in self.fieldlen.items():
            self.feature_len += value
            self.field_start[key] = i
            i += value

    def data_generator(self, dataset_type, epoch=1):
        i = 0
        while i < epoch:
            x = []
            y = []
            filelen = 0
            with open(self.data_info[dataset_type]['path']) as file:
                for line in file.readlines():
                    line = line.strip().split()
                    line_x, line_y = self.convert_line(line, dataset_type=dataset_type)
                    x.append(line_x)
                    y.append(line_y)
                    filelen += 1
                    if (filelen % self.batch_size == 0) or (filelen == self.data_info[dataset_type]['len']):
                        if dataset_type == 'train':
                            yield np.array(x), np.array(y)[:, np.newaxis], i
                        else:
                            yield np.array(x)
                        x = []
                        y = []
            i += 1

    def convert_line(self, line, dataset_type):
        x = np.zeros([self.feature_len])
        y = 0
        for i in range(len(line)):
            field = self.data_info[dataset_type]['cols'][i]

            if field in self.discrete_col:
                dis_feature_idx = self.field_start[field] + self.ledict[field][line[i]]
                x[dis_feature_idx] = 1

            if field in self.numerical_col:
                num_feature_idx = self.field_start[field]
                x[num_feature_idx] = line[i]

            if field in self.targetcol:
                if self.target_type == 'numberical':
                    y = line[i]
                if self.target_type == 'discrete':
                    y = self.targetdict[line[i]]

            if field in self.unusecol:
                pass

        return x, y

    def feature_num(self):
        return self.feature_len