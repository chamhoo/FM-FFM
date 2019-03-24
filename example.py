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
        'cols': cols,
        'discrete_col': singlecatecol,
        'multi_dis_col': multicatecol,
        'uselesscol': uselesscol
    }

    fm_params = {
        'k': 5,
        'l2': 0.,
        'learning_rate': 0.1,
        'loss': 'mse',
        'optimizer': 'gd'
    }

    linear_params = {
        'loss': 'mse',
        'learning_rate': 0.1,
        'optimizer': 'gd'}

    cv_params = {
        'epoch': 5,
        'early_stopping': True,
        'verbose': True,
        'nfolds': 5,
        'early_stopping_epoch': 1}

    ctr = CTR()
    ctr.load_param(**load_params)
    ctr.FM(**fm_params)
    ctr.cv(**cv_params)
