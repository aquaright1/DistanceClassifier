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

    # TODO: make subplots

    for c, dists in distances.items():
        dist_params = params[c]

        dists = np.sort(dists)
        empirical_pdf = dists / dists.size
        gamma_cdf = get_emp_p(dists, dist_params[0], dist_params[1])

        plt.title(f'Class {c}')

        plt.plot(dists, gamma_cdf, label='Gamma CDF')
        plt.plot(dists, empirical_pdf.cumsum(), label='Empirical CDF')
        plt.legend()

        plt.savefig(f'graphs/class_{c}.png')
        plt.show()

if __name__ == '__main__':
    # this code may or not be mostly stolen from `testing.py`
    data= np.genfromtxt('abalone.data', delimiter=",")

    #add preprocessing function
    x = data[:, 1:-1]
    y = data[:, -1]

    model = classifier.NNClassifier()
    model.fit(x, y)
    graph(model)
