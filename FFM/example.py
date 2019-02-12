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
        'batch_size': 4
    }
    data = LoadData(**param)
    data_generator = data.batchload()