import warnings
warnings.filterwarnings("ignore")

from loaddata import *


if __name__ == '__main__':
    param = {
        'path': '../data/ml-100k/u.data',
        'targetcol': ['score'],
        'target_type': 'numberical',
        'cols': ['user', 'item', 'score', 'timestamp'],
        'discrete_col': ['user', 'item'],
        'unusecol': ['timestamp'],
        'batch_size': 64
    }
    data = LoadData(**param)
    x = data.input()
    print(x.shape)
