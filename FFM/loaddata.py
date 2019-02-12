"""
这个 loaddata 文件用于载入 csv 格式的文件转化为 FFM 所需的格式。

"""
import numpy as np
import time


class LoadData(object):
    def __init__(self,
                 path,
                 targetcol, target_type,
                 batch_size=None,
                 cols=[], unusecol=[], discrete_col=[], numerical_col=[]):
        self.path = path
        self.batch_size = batch_size
        self.targetcol = targetcol
        self.target_type = target_type
        self.cols = cols
        self.len = 0
        self.unusecol = unusecol
        self.discrete_col = discrete_col
        self.numerical_col = numerical_col
        self.usecol = self.discrete_col + self.numerical_col
        self.ledict = dict(zip(self.usecol, [dict() for i in self.usecol]))
        self.targetdict = dict()
        self.target_nunique = 0
        self.fieldlen = dict(zip(self.usecol, [0 for i in self.usecol]))
        self.field_start = dict()
        self.feature_len = 0

    def preload(self):
        """
        preload 的作用仅仅是计算出 fieldlen
        """
        with open(self.path) as file:
            a = 0
            for line in file.readlines():
                line = line.strip().split()
                i = 0
                for feature in line:
                    field = self.cols[i]

                    if field in self.targetcol:
                        if self.target_type == 'numberical':
                            pass
                        if self.target_type == 'discrete':
                            isexit = self.targetdict.get(feature, None)
                            if isexit == None:
                                self.targetdict[feature] = self.target_nunique
                                self.target_nunique += 1
                            else:
                                pass

                    if field in self.unusecol:
                        pass

                    if field in self.numerical_col:
                        self.ledict[field] = 0
                        self.fieldlen[field] = 1

                    if field in self.discrete_col:
                        isexit = self.ledict[field].get(feature, False)
                        if isexit is False:
                            self.ledict[field][feature] = self.fieldlen[field]
                            self.fieldlen[field] += 1
                            a += 1

                    i += 1
                self.len += 1

    def convert_line(self, line):
        x = np.zeros([self.feature_len])
        for i in range(len(line)):
            field = self.cols[i]

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

    def batchload(self):
        self.preload()
        x = []
        y = []
        i = 0
        filelen = 0

        for key, value in self.fieldlen.items():
            self.feature_len += value
            print(self.fieldlen)
            self.field_start[key] = i
            i += value

        with open(self.path) as file:
            for line in file.readlines():
                line = line.strip().split()
                line_x, line_y = self.convert_line(line)
                x.append(line_x)
                y.append(line_y)
                filelen += 1
                if (filelen % self.batch_size == 0) or (filelen == self.len):
                    yield np.array(x), np.array(y)
                    x = []
                    y = []










