from load import *
from params import *
from time import time


load_params = {
    'delimiter': {'field': ',', 'multi': ';'},
    'trainpath': trainpath,
    'testpath': testpath,
    'target_type': 'numerical',
    'targetcol': target,
    'split_percentage': 80,
    'batch_size': 4,
    'numerical_col': numcol,
    'cols':cols,
    'discrete_col': singlecatecol,
    'multi_dis_col': multicatecol,
    'uselesscol': uselesscol
}

data = LoadData(**load_params)

for idx, val, y in data.data_generator('train'):
    print(idx)
