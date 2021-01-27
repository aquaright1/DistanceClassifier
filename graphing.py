import scipy as sp
import numpy as np

import classifier

from matplotlib import pyplot as plt

def get_raw_p(array):
    return np.argsort(array)/(len(array) + 1)

def get_emp_p(array, k: float, theta: float):
    dist = sp.stats.gamma(theta, scale = k)
    return dist.cdf(array)

def graph(model):
    params = model.get_params()
    distances = model.get_distances()

    print('params:', params)
    print()
    print('distances:', distances)

if __name__ == '__main__':
    # this code may or not be mostly stolen from `testing.py`
    data= np.genfromtxt('abalone.data', delimiter=",")

    #add preprocessing function
    x = data[:, 1:-1]
    y = data[:, -1]

    model = classifier.NNClassifier()
    model.fit(x, y)
    graph(model)
