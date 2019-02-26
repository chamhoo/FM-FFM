import warnings
warnings.filterwarnings("ignore")

from loaddata import *
from ffm


if __name__ == '__main__':
    data_param = {
        'trainpath': '../data/ml-100k/u.data',
        'testpath': '../data/ml-100k/u5.test',
        'targetcol': 'score',
        'target_type': 'numberical',
        'cols': ['user', 'item', 'score', 'timestamp'],
        'discrete_col': ['user', 'item'],
        'unusecol': ['timestamp'],
        'batch_size': 4
    }

    input = LoadData(**dataparam)
    train = input.data_generator('train')
    test = input.data_generator('test')

    training_param = {}

    try:
