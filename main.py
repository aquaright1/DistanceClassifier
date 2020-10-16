import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import operator
import scipy as sp
from sklearn import preprocessing
import json
from Distance_Classifier import Distance_classifier
from sklearn.metrics import r2_score

def get_raw_p(array):
    return np.argsort(array)/len(array)

def get_emp_p(array, k, theta):
    dist = sp.stats.gamma(theta, scale = k)
    return dist.cdf(array)

LOG = True
FULL = True
ORGANISM = "AT"
NUM_TO_NAME = {
    0: "ER",
    1: "ERDD",
    2: "GEO",
    3: "GEOGD",
    4: "HGG",
    5: "SF",
    6: "SFDD",
    7: "Sticky"
}


data_location_AT = [r"D:\Storage\Research\data\ATER",
                 r"D:\Storage\Research\data\ATERDD",
                 r"D:\Storage\Research\data\ATGEO",
                 r"D:\Storage\Research\data\ATGEOGD",
                 r"D:\Storage\Research\data\ATHGG",
                 r"D:\Storage\Research\data\ATSF",
                 r"D:\Storage\Research\data\ATSFDD",
                 r"D:\Storage\Research\data\ATSticky",
                 r"D:\Storage\Research\data\ATOriginal"]

data_location_CE = [r"D:\Storage\Research\data\CEER",
                 r"D:\Storage\Research\data\CEERDD",
                 r"D:\Storage\Research\data\CEGEO",
                 r"D:\Storage\Research\data\CEGEOGD",
                 r"D:\Storage\Research\data\CEHGG",
                 r"D:\Storage\Research\data\CESF",
                 r"D:\Storage\Research\data\CESFDD",
                 r"D:\Storage\Research\data\CESticky",
                 r"D:\Storage\Research\data\CEOriginal"]

data_location_DM = [r"D:\Storage\Research\data\DMER",
                 r"D:\Storage\Research\data\DMERDD",
                 r"D:\Storage\Research\data\DMGEO",
                 r"D:\Storage\Research\data\DMGEOGD",
                 r"D:\Storage\Research\data\DMHGG",
                 r"D:\Storage\Research\data\DMSF",
                 r"D:\Storage\Research\data\DMSFDD",
                 r"D:\Storage\Research\data\DMSticky",
                 r"D:\Storage\Research\data\DMOriginal"]

data_location_EC = [r"D:\Storage\Research\data\ECER",
                 r"D:\Storage\Research\data\ECERDD",
                 r"D:\Storage\Research\data\ECGEO",
                 r"D:\Storage\Research\data\ECGEOGD",
                 r"D:\Storage\Research\data\ECHGG",
                 r"D:\Storage\Research\data\ECSF",
                 r"D:\Storage\Research\data\ECSFDD",
                 r"D:\Storage\Research\data\ECSticky",
                 r"D:\Storage\Research\data\ECOriginal"]

data_location_HS = [r"D:\Storage\Research\data\HSER",
                 r"D:\Storage\Research\data\HSERDD",
                 r"D:\Storage\Research\data\HSGEO",
                 r"D:\Storage\Research\data\HSGEOGD",
                 r"D:\Storage\Research\data\HSHGG",
                 r"D:\Storage\Research\data\HSSF",
                 r"D:\Storage\Research\data\HSSFDD",
                 r"D:\Storage\Research\data\HSSticky",
                 r"D:\Storage\Research\data\HSOriginal"]

data_location_RN = [r"D:\Storage\Research\data\RNER",
                 r"D:\Storage\Research\data\RNERDD",
                 r"D:\Storage\Research\data\RNGEO",
                 r"D:\Storage\Research\data\RNGEOGD",
                 r"D:\Storage\Research\data\RNHGG",
                 r"D:\Storage\Research\data\RNSF",
                 r"D:\Storage\Research\data\RNSFDD",
                 r"D:\Storage\Research\data\RNSticky",
                 r"D:\Storage\Research\data\RNOriginal"]

data_location_SC = [r"D:\Storage\Research\data\SCER",
                 r"D:\Storage\Research\data\SCERDD",
                 r"D:\Storage\Research\data\SCGEO",
                 r"D:\Storage\Research\data\SCGEOGD",
                 r"D:\Storage\Research\data\SCHGG",
                 r"D:\Storage\Research\data\SCSF",
                 r"D:\Storage\Research\data\SCSFDD",
                 r"D:\Storage\Research\data\SCSticky",
                 r"D:\Storage\Research\data\SCOriginal"]

data_location_SP = [r"D:\Storage\Research\data\SPER",
                 r"D:\Storage\Research\data\SPERDD",
                 r"D:\Storage\Research\data\SPGEO",
                 r"D:\Storage\Research\data\SPGEOGD",
                 r"D:\Storage\Research\data\SPHGG",
                 r"D:\Storage\Research\data\SPSF",
                 r"D:\Storage\Research\data\SPSFDD",
                 r"D:\Storage\Research\data\SPSticky",
                 r"D:\Storage\Research\data\SPOriginal"]

num = 8
cats = num if num <= 8 else 8

ORGANISM = "disregard"

X = []
y = []
for i in range(cats):
        x = pd.read_csv(data_location_CE[i], header = None, sep = ' ').iloc[:,:].values
        for b in x:
            X.append(b)
            y.append(i)
x = X

X = normalize(X)
x_train, x_test, y_train, y_test = train_test_split(X,y)
    # for full test just use X and y
if FULL:
    test_class = Distance_classifier(x,list(y), model = "gamma", threshold = 1/len(x_train))
else:
    test_class = Distance_classifier(x_train, list(y_train))

test_class.fit()


gamma_alphas = test_class.get_gamma_alphas()
details = test_class.get_details()

for i in details.keys():
    details[i] = np.asarray(details[i][i])

actual_p = {}
distri_p = {}

for cat, dist in details.items():

    actual_p[cat] = get_raw_p(np.sort(dist))
    distri_p[cat] = get_emp_p(np.sort(dist), gamma_alphas[cat,1], gamma_alphas[cat,0])

for cat in actual_p.keys():
    for cdf in ["actual", "theory"]:
        np.savetxt(f"{ORGANISM}_{NUM_TO_NAME[cat]}_{cdf}.txt", actual_p[cat] if cdf == "actual" else distri_p[cat])

        #plot the distributions
    plt.plot(actual_p[cat], distri_p[cat])

    if LOG:
        plt.yscale('log')
        plt.xscale('log')

    #Label axis
    plt.xlabel("Emprical CDF")
    plt.ylabel("Theoretical CDF")

    #put r^2 of line
    plt.text(0,1, f"Has a r^2 of {r2_score(actual_p[cat],distri_p[cat])}")

    #plot y = x
    plt.plot([0,1], [0,1])

    plt.savefig(f"{ORGANISM}_log_{NUM_TO_NAME[cat]}.png")

    plt.clf()
