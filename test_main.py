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
from new_distance_classifier import Distance_classifier
from sklearn.metrics import r2_score, accuracy_score

#data file should be in txt format with comma as delimiter and last column is target values.
#If delimiter is not a comma change the delimiter accordingly
FILENAME=r'C:\Users\rione\Documents\UCI\Machine learning research\data_banknote_authentication.txt' #Enter filename here
data=np.genfromtxt(FILENAME,delimiter=",")

print(data.shape)# to make sure everything loaded
X=data[:,0:data.shape[1]-1]
y=data[:,data.shape[1]-1]
X = normalize(X)
x_train, x_test, y_train, y_test = train_test_split(X,y)

def get_raw_p(array):
    return np.argsort(array)/len(array)

def get_emp_p(array, k, theta):
    dist = sp.stats.gamma(theta, scale = k)
    return dist.cdf(array)

LOG = True
FULL = False

# for full test just use X and y
if FULL:
    test_class = Distance_classifier(X,list(y), model = "gamma", threshold = 1/len(x_train))
else:
    test_class = Distance_classifier(x_train, list(y_train),model="gamma", threshold=1/len(x_train))

test_class.fit()

test_class.mle()

gamma_alphas = test_class.get_gamma_alphas()
details = test_class.get_details()

for i in details.keys():
    details[i] = np.asarray(details[i][i])

actual_p = {}
distri_p = {}
for cat, dist in details.items():
    cat=int(cat)
    actual_p[cat] = get_raw_p(np.sort(dist))
    distri_p[cat] = get_emp_p(np.sort(dist), gamma_alphas[cat,1], gamma_alphas[cat,0])

# sort list of pairs with emperical.
    #plot the distributions
    plt.plot(actual_p[cat], distri_p[cat])

    if LOG:
        plt.yscale('log')
        plt.xscale('log')

    #Label axis
    plt.xlabel("Emprical CDF")
    plt.ylabel("Theoretical CDF")

    #put r^2 of line
#     plt.text(0,1, f'Has a r^2 of {r2_score(actual_p[cat],distri_p[cat])}',transform=plt.transAxes)
#     plt.text(1,1,'Pre', transform=ax3.transAxes)
    #plot y = x
    print(f'{cat} has a r^2 of {r2_score(actual_p[cat],distri_p[cat])}')
    plt.plot([0,1], [0,1])
    plt.show()
#     plt.savefig(f"{ORGANISM}_log_{NUM_TO_NAME[cat]}.png")

    plt.clf()
predicted=[]
for point in x_test:
    predicted.append(test_class.predict(point,explicit=False))
print(accuracy_score(y_test,predicted))
#     print("distri",distri_p[cat])
#     print("actual",actual_p[cat])