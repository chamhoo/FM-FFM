"""
 In this example training tasks, we use movielens 100k as training data.
 The movielens 100k's description file location is in ../data/ml-100k/README
 Suggested path = '../data/ml-100k/u.data'
 The col name is [user, movie, score, Timestamp]
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataReader(object):
    def __init__(self, path):
        self.path = path

    def encoder(self):
        self.nunique_dic = dict()
        for subcol in self.discrete_col:
            le = LabelEncoder()
            self.data_dict[subcol] = le.fit_transform(self.data_dict[subcol]).tolist()
            self.nunique_dic[subcol] = max(self.data_dict[subcol]) + 1

    def empty_dict(self):
        dic = dict()
        for col in self.cols:
            dic[col] = []
        return dic

    def input(self, targetcol, cols=None, unusecol=[], discrete_col=None):
        """
        If the first line of csv data is row name, cols = None
        Add the label that needs to be one-hot encoded to the discrete_col
        Targetcol is a list of
        """
        self.cols = cols
        self.targetcol = targetcol
        self.unusecol = unusecol
        self.discrete_col = discrete_col
        self.numerical_col = list(set(self.cols) - set(self.unusecol + self.targetcol) - set(self.discrete_col))

        if self.cols != None:
            self.data_dict = self.empty_dict()

        i = 0; a = 0
        with open(self.path) as file:
            for line in file.readlines():
                line = line.strip().split()
                if (self.cols == None) & (i == 0):
                    self.cols = line
                    self.data_dict = self.empty_dict()
                    a += 1
                else:
                    for sub in range(len(self.cols)):
                        self.data_dict[self.cols[sub]].append(line[sub])
                i += 1
        self.row = i-a

        self.encoder()

        start_col = 0; column = []; value=[]
        for subcol in cols:
            if subcol in self.unusecol:
                pass
            if subcol in self.targetcol:
                y = np.array(self.data_dict[subcol])
            if subcol in self.numerical_col:
                value += self.data_dict[subcol]
                column += [start_col] * self.row
                start_col += 1
            if subcol in self.discrete_col:
                value += [1] * self.row
                column += [i+start_col for i in self.data_dict[subcol]]
                start_col += self.nunique_dic[subcol]
            del self.data_dict[subcol]

        row = [i for i in range(self.row)] * len(self.numerical_col+self.discrete_col)
        x = np.zeros([self.row, start_col])
        for row, col, val in zip(row, column, value):
            x[row, col] += val

        return x, y


if __name__ == '__main__':
    param = {
        'cols': ['user', 'item', 'score', 'timestamp'],
        'targetcol': ['score'],
        'unusecol': ['timestamp'],
        'discrete_col': ['user', 'item']
    }

    data = DataReader('../data/ml-100k/u.data')
    x, y = data.input(**param)

