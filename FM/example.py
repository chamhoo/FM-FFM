import warnings
warnings.filterwarnings("ignore")

from loaddata import *
from FM import *

if __name__ == '__main__':
    BATCH_SIZE = 32
    EPOCH = 3

    data_param = {
        'trainpath': '../data/ml-100k/u.data',
        'testpath': '../data/ml-100k/u5.test',
        'targetcol': 'score',
        'target_type': 'numberical',
        'cols': ['user', 'item', 'score', 'timestamp'],
        'discrete_col': ['user', 'item'],
        'unusecol': ['timestamp'],
        'batch_size': BATCH_SIZE
    }

    input = LoadData(**data_param)
    train = input.data_generator('train', epoch=EPOCH)
    test = input.data_generator('test')
    feature_num = input.feature_num()

    try:
        fm = FM()
        fm.traindata(train)
        fm.predictdata(test)
        fm.info_reseive(col=feature_num)
        fm.model(k=10, l2=0., learning_rate=0.01, loss_name='mse')
        fm.train()
        y_pre = fm.predict()
    finally:
        fm.close()