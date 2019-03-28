import time
import numpy as np
import matplotlib.pyplot as plt
from CTR import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import warnings
warnings.filterwarnings('ignore')


class AutoTuning(CTR):
    def cv_score(self, params):
        param = {}
        for key, value in self.model_params.items():
            param[key] = value
        for key, value in params.items():
            param[key] = value
        self.model(**param)
        return self.cv(**self.cv_params)

    def f(self, params):
        self.eval += 1
        score = self.cv_score(params)['mean_score']
        self.fmin_recorder['score'].append(score)
        for key, value in params.items():
            self.fmin_recorder['param'][key].append(value)
        if self.fmin_verbose == 1:
            if self.best_score > score:
                self.best_score = score
                print(f'new best, eval {self.eval}, score {self.best_score}, param {params}')

        if self.fmin_verbose == 2:
            print(f'eval {self.eval}, score {score}, param {params}')

        if self.fmin_verbose == 3:
            num_params = len(params)
            col = np.ceil(np.sqrt(num_params))
            row = np.floor(np.sqrt(num_params))
            for i, [key, value] in enumerate(self.fmin_recorder['param'].items()):
                self.ax[key] = self.fig.add_subplot(row, col, i+1)
                self.ax[key].cla()
                self.ax[key].scatter(value, self.fmin_recorder['score'])
            plt.pause(0.01)

        else:
            pass
        return {'loss': score, 'status': STATUS_OK}

    def fmin(self, model, space_dict, model_params, cv_params, verbose=0, max_evals=100):
        """
        We use Hyperopt to achieve tunnning automation.
        you can use [pip install hyperopt] command to install this package.
        - https://github.com/hyperopt/hyperopt
        :param model: <class 'method'>, a model function
        :param space_dict: dict, record parameter search space
        :param model_params: dict, the model parameter what doesn't use in space_dict.
        :param verbose: [0, 1, 2, 3] (default=1), 0 = almost silent, Only one line of progress bar.
               1 = Update only when better parameters appear, 2 = Update every time,
               3 = Update and plot every time.
        :param cv_params: dict, The cv parameter.
        :param max_evals: int, The maximum number of parameter searches you can afford,
               the more likely you are to search for better parameters, but the longer it takes.
        :return: best training param
        """
        self.best_score = 2**32
        self.eval = 0
        self.model = model
        self.cv_params = cv_params
        self.fmin_verbose = verbose
        self.model_params = model_params
        self.fmin_recorder = dict()
        self.fmin_recorder['score'] = []
        self.fmin_recorder['param'] = dict(zip(space_dict.keys(), [[] for i in range(len(space_dict))]))

        if self.fmin_verbose == 3:
            self.ax = {}
            self.fig = plt.figure()

        trials = Trials()
        best = fmin(self.f, space_dict, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        plt.close()
        return best
