"""
In this file, we load the data from the target csv file and turn it into generator.
We used an innovative preloading method that traversed the file, Label-Encode,
and statistics basic information, such as the number of rows, before the file was officially read.

The benefits of this treatment are:
1. Save memory, because the preloading uses Label-Encoder,
 so the loading process can take up very little memory.
2. Very fast, We will compress the time of loading data as much as possible.

Input parameter description:
trainpath - Training data path
testpath - Test data path
target_type - 'discrete' or 'numerical'
targetcol - The name of the target column
split_percentage - The training set accounts for the proportion of the total input samples,
                   and we will divide the training set and validation set accordingly.
batch_size - batch-size
numerical_col - List of column names of all numerical data
cols - List of column names of all data
discrete_col - List of column names of all discrete data
multi_dis_col - List of column names of all multi-discrete data
uselesscol - List of column names of all useless data

Warning: The loaded csv file needs to meet a certain format:
the delimiter must be different from the multi-valued discrete field.

author: leechh
"""


import numpy as np
from time import time


class LoadData(object):
    def __init__(self,
                 trainpath,
                 testpath,
                 target_type,
                 targetcol,
                 delimiter,
                 split_percentage=100,
                 batch_size=64,
                 numerical_col=[],
                 cols=[], discrete_col=[], multi_dis_col=[], uselesscol=[]):

        self.delimiter = delimiter
        self.seed = np.random.randint(0, 200, 1)[0]
        self.batch_size = batch_size
        self.split_percentage = split_percentage
        self.target_type = target_type
        self.numerical_col = numerical_col
        self.targetcol = targetcol
        self.discrete_col = discrete_col
        self.multi_dis_col = multi_dis_col
        self.uselesscol = uselesscol
        self.usecol = self.discrete_col + self.numerical_col + self.multi_dis_col

        self.ledict = dict(zip(self.usecol, [dict() for i in self.usecol]))
        self.fieldlen = dict(zip(self.usecol, [0 for i in self.usecol]))
        self.multi_max = dict(zip(self.multi_dis_col, [0 for i in self.multi_dis_col]))

        self.targetdict = dict()
        self.field_start = dict()
        self.num_feature = 0
        self.target_nunique = 0
        self.random_cate = []

        self.idx = []
        self.x_idx = []
        self.x_val = []
        self.y = []
        self.maxlen = 0

        testcols = cols.copy()
        testcols.remove(self.targetcol)

        self.datainfo = {
            'train': {'path': trainpath, 'cols': cols, 'len': 0},
            'test': {'path': testpath, 'cols': testcols, 'len': 0}
        }
        self._preload()

    def reset_seed(self, seed):
        """
        You can set the seed to a fixed value.
        """
        self.seed = seed

    def _preload(self):
        for _type in ['train', 'test']:

            with open(self.datainfo[_type]['path']) as file:
                for line in file.readlines():
                    line = line.strip().split(self.delimiter['field'])
                    for idx in range(len(line)):
                        field = self.datainfo[_type]['cols'][idx]

                        if field in self.numerical_col:
                            self.ledict[field][field] = 0
                            self.fieldlen[field] = 1

                        elif field in self.discrete_col:
                            is_exit = self.ledict[field].get(line[idx], False)
                            if is_exit is False:
                                self.ledict[field][line[idx]] = self.fieldlen[field]
                                self.fieldlen[field] += 1

                        elif field in self.multi_dis_col:
                            self.multi_max[field] = max(self.multi_max[field], len(line[idx]))
                            for i in line[idx].replace(' ', '').split(self.delimiter['multi']):
                                is_exit = self.ledict[field].get(i, False)
                                if is_exit is False:
                                    self.ledict[field][i] = self.fieldlen[field]
                                    self.fieldlen[field] += 1

                        elif field is self.targetcol:
                            if self.target_type == 'discrete':
                                isexit = self.targetdict.get(line[idx], False)
                                if isexit is False:
                                    self.targetdict[line[idx]] = self.target_nunique
                                    self.target_nunique += 1

                        else:
                            pass
                    self.datainfo[_type]['len'] += 1

        i = 0
        for key, value in self.fieldlen.items():
            self.field_start[key] = i
            self.num_feature += value
            i += value

    def _col_transform(self, filelen, line, subtype):
        for i in range(len(line)):
            field = self.datainfo[subtype]['cols'][i]

            if field in self.numerical_col:
                field_index = self.usecol.index(field)
                self.idx.append([filelen, field_index, 0])
                self.x_idx.append(int(self.ledict[field][field] + self.field_start[field]))
                self.x_val.append(float(line[i]))

            elif field in self.discrete_col:
                field_index = self.usecol.index(field)
                self.idx.append([filelen, field_index, 0])
                self.x_idx.append(int(self.ledict[field][line[i]] + self.field_start[field]))
                self.x_val.append(1)

            elif field in self.multi_dis_col:
                num = 0
                field_index = self.usecol.index(field)
                field_keys = self.ledict[field].keys()
                multi_count = dict(zip(field_keys, np.zeros(len(field_keys))))
                for sub in line[i].replace(' ', '').split(self.delimiter['multi']):
                    multi_count[sub] += 1
                for keys, value in multi_count.items():
                    if value != 0:
                        self.idx.append([filelen, field_index, num])
                        self.x_idx.append(int(self.ledict[field][keys] + self.field_start[field]))
                        self.x_val.append(value)
                        num += 1
                self.maxlen = max(num, self.maxlen)

            elif field in self.targetcol:
                if self.target_type == 'discrete':
                    target_array = np.zeros(self.target_nunique)
                    target_array[self.targetdict[line[i]]] = 1
                    self.y.append(target_array)
                else:
                    self.y.append(line[i])

            else:
                pass

    def _random_split(self):
        """
        Linear congruential generator
        - https://en.wikipedia.org/wiki/Linear_congruential_generator
        """
        m = 2**32
        seed = self.seed
        for i in range(m):
            nextseed = (214013*seed + 2531011) % m
            pp = (self.split_percentage / 100) > (nextseed / m)
            yield pp
            seed = nextseed


    def data_generator(self, dataset_type):
        """
        :param dataset_type: 'train', 'valid' or 'test'
        :return: a dataset generator of this data type
        """
        batch_idx = 0
        size = 0
        split = self._random_split()

        if dataset_type in ['train', 'valid']:
            _type = 'train'
        else:
            _type = 'test'

        with open(self.datainfo[_type]['path']) as file:
            for line in file:
                size += 1
                if ((dataset_type == 'train') == next(split)) or (dataset_type is 'test'):
                    line = line.strip().split(self.delimiter['field'])
                    self._col_transform(batch_idx, line, _type)
                    batch_idx += 1

                    if (batch_idx == self.batch_size) or (size == self.datainfo[_type]['len']):
                        idx_array = np.zeros([batch_idx, len(self.usecol), self.maxlen+1])
                        val_array = idx_array.copy()

                        for i, [x, y, z] in enumerate(self.idx):
                            idx_array[x, y, z] = self.x_idx[i]
                            val_array[x, y, z] = self.x_val[i]

                        if _type == 'train':
                            yield idx_array, val_array, np.array(self.y), batch_idx
                        else:
                            yield idx_array, val_array, batch_idx

                        del idx_array, val_array
                        self.idx.clear()
                        self.x_idx.clear()
                        self.x_val.clear()
                        self.y.clear()
                        self.maxlen = 0
                        batch_idx = 0



