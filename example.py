from auto_tuning import *
from params import *

if __name__ == '__main__':

    load_params = {
        'delimiter': {'field': ',', 'multi': ';'},
        'trainpath': trainpath,
        'testpath': testpath,
        'target_type': 'numerical',
        'targetcol': target,
        'split_percentage': 80,
        'batch_size': 1000,
        'numerical_col': numcol,
        'cols': cols,
        'discrete_col': singlecatecol,
        'multi_dis_col': multicatecol,
        'uselesscol': uselesscol
    }

    fm_params = {
        'k': 5,
        'l2': 0.,
        'learning_rate': 0.001,
        'loss': 'mse',
        'optimizer': 'adam'
    }

    linear_params = {
        'loss': 'mse',
        'learning_rate': 0.001,
        'optimizer': 'adam'}

    cv_params = {
        'epoch': 1000,
        'early_stopping': True,
        'verbose': True,
        'nfolds': 5,
        'early_stopping_epoch': 1}

    train_params = {
        'epoch': 5,
        'early_stopping': True,
        'early_stopping_epoch': 1,
        'retrain': False,
        'verbose': 2
    }

    ctr = CTR()
    ctr.load_param(**load_params)
    t = time()
    a = 0
    for i in ctr.data_generator('train'):
        a += 1
    print(time()-t)
"""
    ctr.linear_regression(**linear_params)
    ctr.cv(**cv_params)
    ctr = AutoTuning()
    ctr.load_param(**load_params)

    fm = ctr.FM

    space_dict = {
        'k': hp.choice('k', range(1, 10))
    }
    
    ctr.fmin(model=fm,
             space_dict=space_dict,
             model_params=fm_params,
             cv_params=cv_params,
             max_evals=30,
             verbose=2)
"""


