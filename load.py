"""
In this file, we load the data from the target csv file and turn it into generator.
We used an innovative preloading method that traversed the file, Label-Encode,
and statistics basic information, such as the number of rows, before the file was officially read.

The benefits of this treatment are:
1. Save memory, because the preloading uses Label-Encoder,
 so the loading process can take up very little memory.
2. Very fast, We will compress the time of loading data as much as possible,

Warning: The loaded csv file needs to meet a certain format:
the delimiter must be different from the multi-valued discrete field.

author: leechh
"""
import numpy as np


class LoadData(object):
    def __init__(self,
                 trainpath,
                 testpath,
                 target_type,
                 targetcol,
                 split_percentage=100,
                 batch_size=64,
                 numerical_col=[],
                 cols=[], discrete_col=[], multi_dis_col=[], unusecol=[]):

        self.seed = np.random.randint(0, 200, 1)[0]
        self.batch_size = batch_size
        self.split_percentage = split_percentage
        self.target_type = target_type
        self.numerical_col = numerical_col
        self.targetcol = targetcol
        self.discrete_col = discrete_col
        self.multi_dis_col = multi_dis_col
        self.unusecol = unusecol
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

        self.datainfo = {
            'train': {'path': trainpath, 'cols': cols, 'len': 0},
            'test':{'path': testpath, 'cols': list(set(cols) - set(targetcol)), 'len': 0}
        }
        self._preload()

    def reset_seed(self, seed):
        self.seed = seed

    def _preload(self):
        for subtype in ['train', 'test']:

            with open(self.datainfo[subtype]['path']) as file:
                for line in file.readlines():
                    line = line.split(',')
                    for idx in range(len(line)):
                        field = self.datainfo[subtype]['cols'][idx]

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
                            for i in line[idx].replace(' ', '').split(';'):
                                is_exit = self.ledict[field].get(i, False)
                                if is_exit is False:
                                    self.ledict[field][i] = self.fieldlen[field]
                                    self.fieldlen[field] += 1

                        elif field == self.targetcol:
                            if self.target_type == 'discrete':
                                isexit = self.targetdict.get(line[idx], False)
                                if isexit is False:
                                    self.targetdict[line[idx]] = self.target_nunique
                                    self.target_nunique += 1

                        else:
                            pass
                    self.datainfo[subtype]['len'] += 1

        i = 0
        for key, value in self.fieldlen.items():
            self.num_feature += value
            self.field_start[key] = i
            i += value

    def col_transform(self, filelen, line, subtype):
        for i in range(len(line)):
            field = self.datainfo[subtype]['cols'][i]
            field_index = self.usecol.index(field)

            if field in self.numerical_col:
                self.idx.append([filelen, field_index, 0])
                self.x_idx.append(self.ledict[field][field] + self.field_start[field])
                self.x_val.append(float(line[i]))

            if field in self.discrete_col:
                self.idx.append([filelen, field_index, 0])
                self.x_idx.append(self.ledict[field][line[i]] + self.field_start[field])
                self.x_val.append(1)

            if field in self.multi_dis_col:
                num = 0
                field_keys = self.ledict[field].keys()
                multi_count = dict(zip(field_keys, np.zeros(len(field_keys))))
                for sub in line[i].replace(' ', '').split(';'):
                    multi_count[sub] += 1
                for keys, value in multi_count:
                    if value != 0:
                        self.idx.append([filelen, field_index, num])
                        self.x_idx.append(self.ledict[field][keys] + self.field_start[field])
                        self.x_val.append(value)
                        num += 1
                self.maxlen = max(num, self.maxlen)

            if field in self.targetcol:
                if self.target_type == 'discrete':
                    self.y.append(self.targetdict[line[i]])
                else:
                    self.y.append(line[i])

    def random_split(self):
        np.random.seed(self.seed)
        for rand in np.random.randint(0, 100, int(1e+7)):
            yield rand

    def data_generator(self, subtype, istrain):
        filelen = 0
        iterations = 0
        istrue = self.random_split()
        with open(self.datainfo[subtype]['path']) as file:
            for line in file.readlines():
                iterations += 1

                if (istrain == (self.split_percentage > next(istrue))) & (subtype == 'train'):
                    line = line.split(',')
                    self.col_transform(filelen, line, subtype)
                    filelen += 1

                    if (filelen % self.batch_size == 0) or (iterations == self.datainfo[subtype]['len']):
                        idx_array = np.zeros([filelen, len(self.usecol), self.maxlen])
                        val_array = idx_array.copy()
                        for i, [x, y, z] in enumerate(self.idx):
                            idx_array[x, y, z] = self.x_idx[i]
                            val_array[x, y, z] = self.x_val[i]

                        if subtype == 'train':
                            yield idx_array, val_array, self.y
                        else:
                            yield idx_array, val_array

                        del idx_array, val_array
                        self.idx = []
                        self.x_idx = []
                        self.x_val = []
                        self.y = []
                        self.maxlen = 0
                        filelen = 0


