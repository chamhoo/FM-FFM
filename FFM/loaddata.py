"""
这个 loaddata 文件用于载入 csv 格式的文件转化为 FFM 所需的格式。

"""
import numpy as np


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
                        self.fieldlen[field] = 0

                    if field in self.discrete_col:
                        isexit = self.ledict[field].get(feature, False)
                        if isexit is False:
                            self.ledict[field][feature] = self.fieldlen[field]
                            self.fieldlen[field] += 1
                            a += 1

                    i += 1
                self.len += 1

    def batchread(self):
        filelen = 0
        batch_list = []
        with open(self.path) as file:
            for line in file.readlines():
                batch_list += line.strip().split()
                filelen += 1
                if (filelen % self.batch_size == 0) & (filelen == self.len):
                    yield batch_list
                    batch_list = []

    def input(self):
        self.preload()
        feature_len = 0
        field_start = dict(); i = 0
        for key,value in self.fieldlen.items():
            feature_len += value + 1
            field_start[key] = i
            i += value + 1

        for batch_list in self.batchread():
            collen = len(self.usecol)
            rowlen = len(batch_list) / len(self.usecol)
            x = np.zeros([rowlen, feature_len])
            y = np.array([])
            col_list = self.cols * rowlen
            row_list = np.hstack([[i]*collen for i in range(rowlen)]).tolist()
            for i in len(batch_list):
                if col_list[i] in self.unusecol:
                    pass
                if col_list[i] in self.targetcol:
                    if self.target_type == 'numberical':
                        y = np.append(y, batch_list[i])
                    if self.target_type == 'discrete':
                        y = np.append(y, self.targetdict[batch_list[i]])
                if col_list[i] in self.numerical_col:
                    x[row_list[i], field_start[col_list[i]]] = batch_list[i]
                if col_list[i] in self.discrete_col:
                    x[row_list[i], field_start[col_list[i]]+self.ledict[col_list[i]][batch_list]] = 1
                    print(y)
        return x, y










