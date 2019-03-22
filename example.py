from time import time

from CTR import *
from params import *

if __name__ == '__main__':

    load_params = {
        'delimiter': {'field': ',', 'multi': ';'},
        'trainpath': trainpath,
        'testpath': testpath,
        'target_type': 'numerical',
        'targetcol': target,
        'split_percentage': 80,
        'batch_size': 32,
        'numerical_col': numcol,
        'cols':cols,
        'discrete_col': singlecatecol,
        'multi_dis_col': multicatecol,
        'uselesscol': uselesscol
    }

    fm_params = {
        'k': 10,
        'l2': 0.,
        'learning_rate': 0.01,
        'loss': 'mse',
        'optimizer': 'gd'
    }

    linear_params = {
        'learning_rate': 0.01,
        'loss': 'mse',
        'optimizer': 'gd'}

    train_params = {
        'epoch': 10,
        'early_stopping': True,
        'verbose': True,
        'early_stopping_epoch': 1,
        'save_recorder': False
    }

    ctr = CTR()
    ctr.load_param(**load_params)
    for i in range(5):
        a = 0
        print('-----------')
        for val in ctr.data_generator('train'):
            a += 1
            if val[2].shape[0] != 32:
                print('train', a, val[2].shape)
                print('ff', ctr.idx, ctr.x_idx, ctr.x_val, ctr.y, ctr.maxlen)
        a = 0
        for val in ctr.data_generator('valid'):
            a += 1
            if val[2].shape[0] != 32:
                print('valid', a, val[2].shape)
    """
    ctr = CTR()
    ctr.load_param(**load_params)
    ctr.linear_regression(**linear_params)
    try:
        ctr.train(**train_params)
    finally:
        ctr.close()
    """