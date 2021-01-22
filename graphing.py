import scipy as sp
import numpy as np

def get_raw_p(array):
    return np.argsort(array)/(len(array) + 1)

def get_emp_p(array, k: float, theta: float):
    dist = sp.stats.gamma(theta, scale = k)
    return dist.cdf(array)

def graph(model):
    pass
