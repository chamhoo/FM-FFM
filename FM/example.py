from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from loaddata import *
from FM import *

if __name__ == '__main__':
    param = {
        'cols': ['user', 'item', 'score', 'timestamp'],
        'targetcol': ['score'],
        'unusecol': ['timestamp'],
        'discrete_col': ['user', 'item']
    }

    data = DataReader('../data/ml-100k/u.data')
    x, y = data.input(**param)
    print(x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test)
    del x, y
    try:
        fm = FM()
        fm.traindata(x_train, y_train)
        fm.validdata(x_valid, y_valid)
        fm.predictdata(x_test)
        fm.model(k=10, l2=0., learning_rate=0.01, loss_name='mse')
        fm.train(epochs=3,batch_size=128)
        y_pre = fm.predict()
    finally:
        fm.close()