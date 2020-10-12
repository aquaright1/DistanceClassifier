import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.neighbors import RadiusNeighborsClassifier
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import operator
import scipy as sp
from sklearn import preprocessing
import json
from Distance_Classifier import Distance_classifier, distance_stack
from sklearn.metrics import r2_score


def OCD_path(id):
    return f'OCD-CON\\OCD\\a{id}_fc_mat.txt'

def CON_path(id):
    return f'OCD-CON\CON1\c{id}_fc_mat.txt'

def set_above(values, boundary):
    return (values > boundary).astype(int)

def percent_above(values, percentile):
    return (values > np.percentile(values, percentile)).astype(int)

def return_set(boundary):
    def above(values):
        return (values > boundary).astype(int)

    return above

OCD_ids = [113, 130, 146, 154, 159, 162, 164, 168, 215, 217, 222, 225, 226, 227, 228, 229, 230, 233, 235, 236, 237]
baselines = np.asarray([.5,.6,.65,.7,.75,.8,.85,.9])
# baselines *= 100
averages = []
accuracies = []

models = []
functs = []

X = []
y = []
for id in OCD_ids:
    x = pd.read_csv(OCD_path(id), header = None, sep = '\n').iloc[:,:].values
    values = np.array([])
    for b in x:
        # print(np.append(values,b, axis = 0))
        values = np.append(values, b)


    X.append(values)
    y.append(1)

CON_ids = [105, 107, 112, 118, 119, 136, 137, 139, 148, 156, 161, 163, 171, 173, 174, 209, 211, 213, 219, 224]
for id in CON_ids:
    x = pd.read_csv(CON_path(id), header = None, sep = '\n').iloc[:,:].values
    values = np.array([])
    for b in x:
        # print(np.append(values,b, axis = 0))
        values = np.append(values, b)

    X.append(values)
    y.append(0)

for baseline in baselines:

    X = np.asarray(X)
    y = np.asarray(y)

    loo = LeaveOneOut()
    acc = []
    scores = []
    no_class = 0
    # for train, test in loo.split(X):
    #     # print(f"train is {train} and test is {test}")
    #     X_train, X_test = X[train], X[test]
    #     y_train, y_test = y[train], y[test]

    test_class = Distance_classifier(X,list(y), model = "gamma")

        # test_class.fit()
        #
        # test_class.mle()
        # # print(y_test[0], test_class.predict(X_test, explicit = False))
        # predict = test_class.predict(X_test, explicit = False)
        # # print(f"total score is : {test_class.score(explicit = True)}")
        # scores.append(test_class.score(explicit = False))
        # if y_test[0] == predict:
        #     acc.append(1)
        # else:
        #     acc.append(0)
        #     if predict == -1:
        #         no_class += 1

    averages.append(np.mean(scores))
    accuracies.append(np.mean(acc))

    models.append(test_class)

    test = return_set(baseline)
    functs.append(test)

    # print(f"average score is {np.mean(scores)} with {np.mean(acc)} accuracy")

stack = distance_stack(models, functs)
# stack.fit(X,y)

# plt.plot(baselines, averages, c =  "red")
# plt.plot(baselines, accuracies, c = "blue")
# plt.show()

acc = []
test1 = lambda x: (x > .5).astype(int)
for train, test in loo.split(X):
    # print(f"train is {train} and test is {test}")
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    stack.fit(X_train, y_train)
    # print(X_train, test1(X_train))

    # print(y_test[0], test_class.predict(X_test, explicit = False))
    predict = stack.predict(X_test)
    # print(f"total score is : {test_class.score(explicit = True)}")
    # scores.append(test_class.score(explicit = False))
    if y_test[0] == predict:
        acc.append(1)
    else:
        acc.append(0)
        if predict == -1:
            no_class += 1
print(f"stack accuracy is {sum(acc)/len(acc)}")
