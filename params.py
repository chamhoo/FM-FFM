"""
author: leechh
"""

import numpy as np


"""
trainpath = './data/elo/train100.csv'
testpath = './data/elo/test100.csv'

cols = np.load('./data/elo/cols.npy').tolist()
target = np.load('./data/elo/target.npy').tolist()[0]
numcol = np.load('./data/elo/numcol.npy').tolist()
singlecatecol = np.load('./data/elo/singlecatecol.npy').tolist()
multicatecol = np.load('./data/elo/multicatecol.npy').tolist()
uselesscol = np.load('./data/elo/unusecol.npy').tolist()


"""
trainpath = './data/ml-100k/train.csv'
testpath = './data/ml-100k/test.csv'

cols = ['user', 'item', 'score', 'time']
target = 'score'
numcol = []
singlecatecol = ['user', 'item']
multicatecol = []
uselesscol = ['time']