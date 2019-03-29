"""
In this file, we load the data from the target csv file and turn it into generator.
We used an innovative preloading method that traversed the file, Label-Encode,
and statistics basic information, such as the number of rows, before the file was officially read.

The benefits of this treatment are:
1. Save memory, because the preloading uses Label-Encoder,
 so the loading process can take up very little memory.
2. Very fast, We will compress the time of loading data as much as possible.

load parameter description:
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
from time import time
import numpy as np
from queue import Queue
from collections import deque
from multiprocessing import Pool, cpu_count


class LoadData(object):

    def load_param(self,
                   target_type,
                   targetcol,
                   delimiter,
                   numerical_col,
                   cols, discrete_col, multi_dis_col, uselesscol,
                   trainpath=None,
                   testpath=None,
                   split_percentage=100,
                   batch_size=64,):

        self.delimiter = delimiter
        self.seed = 16843009
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
        for _type in [i for i in ['train', 'test'] if self.datainfo[i]['path'] is not None]:

            with open(self.datainfo[_type]['path']) as file:
                for line in file:
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

                        elif field in self.targetcol:
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

    def _col_transform(self, q, filelen, line, datainfo):
        # prepare
        line = line.strip().split(self.delimiter['field'])

        # convert line
        for i, unit in enumerate(line):
            field = datainfo['cols'][i]

            if field in self.numerical_col:
                field_index = self.usecol.index(field)
                q['idx'].append([filelen, field_index, 0])
                q['x_idx'].append(self.ledict[field][field] + self.field_start[field])
                q['x_val'].append(float(unit))

            elif field in self.discrete_col:
                field_index = self.usecol.index(field)
                q['idx'].append([filelen, field_index, 0])
                q['x_idx'].append(self.ledict[field][unit] + self.field_start[field])
                q['x_val'].append(1)

            elif field in self.multi_dis_col:
                num = 0
                field_index = self.usecol.index(field)
                field_keys = self.ledict[field].keys()
                multi_count = dict(zip(field_keys, np.zeros(len(field_keys))))
                for sub in unit.replace(' ', '').split(self.delimiter['multi']):
                    multi_count[sub] += 1
                for keys, value in multi_count.items():
                    if value != 0:
                        q['idx'].append([filelen, field_index, num])
                        q['x_idx'].append(self.ledict[field][keys] + self.field_start[field])
                        q['x_val'].append(value)
                        num += 1
                q['maxlen'].append(num)

            elif field in self.targetcol:
                if self.target_type == 'discrete':
                    target_array = np.zeros(self.target_nunique)
                    target_array[self.targetdict[unit]] = 1
                    q['y'].append(target_array)
                    q['y_idx'].append(filelen)
                else:
                    q['y'].append([float(unit)])
                    q['y_idx'].append(filelen)

    def _pseudo_random(self, a, b, isbool):
        """
        Linear congruential generator
        - https://en.wikipedia.org/wiki/Linear_congruential_generator
        """
        m = 2**32
        seed = self.seed
        while True:
            nextseed = (a*seed + b) % m
            if isbool:
                yield (self.split_percentage / 100) > (nextseed / m)
            else:
                yield nextseed
            seed = nextseed

    def _input_generator(self, dataset_type):
        size, batch_size, chunk = 0, 0, []
        split = self._pseudo_random(214013, 2531011, True)

        # Choose the appropriate file path
        if dataset_type in ['train', 'valid']:
            datainfo = self.datainfo['train']
        else:
            datainfo = self.datainfo['test']

        with open(datainfo['path']) as file:
            for line in file:
                size += 1
                if ((dataset_type == 'train') == next(split)) or (dataset_type is 'test'):
                    chunk.append(line)
                    batch_size += 1
                if (batch_size == self.batch_size) or ((size == datainfo['len']) & (batch_size != 0)):
                    yield chunk, batch_size, datainfo
                    batch_size, chunk = 0, []

    def data_generator(self, dataset_type, numpool=-1):
        """
        :param dataset_type: 'train', 'valid' or 'test'
        :return: a dataset generator of this data type
        """
        # assert
        assert dataset_type in ['train', 'valid', 'test'], 'dataset_type is out of the range'
        assert type(numpool) is int, 'the type of numpool is int'

        # numpool
        if numpool < 1:
            numpool = cpu_count()

        # empty resultdic
        resultdic = dict(zip(
            ['y_idx', 'idx', 'x_idx', 'x_val', 'y', 'maxlen'],
            [deque(), deque(), deque(), deque(), deque(), deque()]))

        for chunk, batch_size, datainfo in self._input_generator(dataset_type):
            # convert chunk
            t = time()
            for i, line in enumerate(chunk):
                self._col_transform(resultdic, i, line, datainfo)
            print(time()-t)
            # find maxlen
            maxlen = 1
            while True:
                try:
                    maxlen = max(maxlen, resultdic['maxlen'].pop())
                except IndexError:
                    break

            # construct output
            array_size = [batch_size, len(self.usecol), maxlen]
            idx_array = np.zeros(array_size, dtype=int)
            val_array = np.zeros(array_size, dtype=float)
            y_list = [0 for i in range(batch_size)]

            while True:
                try:
                    x, y, z = resultdic['idx'].popleft()
                    idx_array[x, y, z] = resultdic['x_idx'].popleft()
                    val_array[x, y, z] = resultdic['x_val'].popleft()
                except IndexError:
                    break

            while True:
                try:
                    y_list[resultdic['y_idx'].popleft()] = resultdic['y'].popleft()
                except IndexError:
                    break

            if y_list[0] == 0:
                yield idx_array, val_array, np.array(y_list), batch_size
            else:
                yield idx_array, val_array, batch_size





