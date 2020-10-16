import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import operator
import scipy as sp
from sklearn import preprocessing
import json
from Distance_Classifier import Distance_classifier
from sklearn.metrics import r2_score
from sklearn.utils import shuffle


def get_raw_p(array):
    return np.argsort(array)/len(array)

def get_emp_p(array, k, theta):
    dist = sp.stats.gamma(theta, scale = k)
    return dist.cdf(array)


def get_data():
    leaf_path = 'leaf.csv'
    df = pd.read_csv(leaf_path)
    df = df.loc[:, df.columns != "Specimen Number"]

    df_y = df[["Class (Species)"]]
    df_X = df.loc[:, df.columns != "Class (Species)"]

    y = df_y.to_numpy().reshape(1, len(df_y))[0]
    X = df_X.to_numpy()
    for index, val in enumerate(y):
        y[index] -= 7 if val > 15 else 1

    X, y = shuffle(X, y, random_state = 40061476)
    X = normalize(X)

    return X, y

def k_folds_test(compare_model = KNeighborsClassifier):
    X, y = get_data()

    kf = KFold(n_splits = 10, shuffle = True, random_state = 2020)
    scores_dc = []
    scores_knn = []

    kn = compare_model()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(train_index, test_index)

        dist_class = Distance_classifier(X_train, y_train)
        dist_class.fit()
        kn.fit(X_train, y_train)
        # print(dist_class.predict(X[15]), dist_class.predict(X[15], explicit = False))
        # print(kn.predict([X[15]]), y[15])
        scores_dc.append(dist_class.score(X_test, y_test))
        scores_knn.append(kn.score(X_test, y_test))


    print(f"Scores dc: {scores_dc}, Avg Score: {np.mean(scores_dc)}")
    print(f"Scores {compare_model}: {scores_knn}, Avg Score: {np.mean(scores_knn)}")

def pdf_compare(data, X, y, log = False):
    dist_class = Distance_classifier(X, y)
    dist_class.fit()

    gamma_alphas = dist_class.get_params()
    details = dist_class.get_details()
    for i in details.keys():
        details[i] = np.asarray(details[i][i])

    actual_p = {}
    distri_p = {}

    for cat, dist in details.items():
        actual_p[cat] = get_raw_p(np.sort(dist))
        distri_p[cat] = get_emp_p(np.sort(dist), gamma_alphas[cat,1], gamma_alphas[cat,0])

        plt.plot(actual_p[cat], distri_p[cat])

        if log:
            plt.yscale('log')
            plt.xscale('log')

        plt.text(0,1, f"Has a r^2 of {r2_score(actual_p[cat],distri_p[cat])}")
        plt.plot([0,1], [0,1])

        plt.savefig(f"pdf comparision for {data} class {cat} log {log}.png")
        plt.clf()

X, y = get_data()
pdf_compare("leaf", X, y, True)
