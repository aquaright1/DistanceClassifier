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

def distance(origin, other):
    return np.sum((origin - other) ** 2)**(1/2)

def read_data(path, header = None, seperator = ' '):
    return pd.read_csv(path, header = header, sep = seperator).iloc[:,:].values

def volume(dimensions, radius):
    return np.pi**(dimensions/2) * radius ** dimensions/ sp.special.gamma(dimensions/2 +1)

def customScaling(distances, scale = (1/3)):
    return distances ** scale

class Distance_classifier():

    def __init__(self, model = "auto"):
        pass

    def __init__(self, X, y, model = "gamma", threshold = .00, kd_tree = False):
        self.kd = kd_tree
        self.data = np.asarray(X)
        self.labels = np.asarray(y)
        self.model = model
        self.threshold = threshold

        #order the data to fit with processing
        proper_order = np.unravel_index(np.argsort(self.labels, axis=None), self.labels.shape)
        self.data = self.data[proper_order]
        self.labels = self.labels[proper_order]

    def distances(self, data):
        zeros = 0
        short_dist = defaultdict(int)
        for i, to_data in enumerate(self.data):
            expect_dist = distance(data, to_data)
            if expect_dist != 0:
                    if short_dist[self.labels[i]] > expect_dist:
                        short_dist[self.labels[i]] = expect_dist
                    if short_dist[self.labels[i]] == 0:
                        short_dist[self.labels[i]] = expect_dist
            elif expect_dist == 0:
                zeros += 1
        if zeros != 1:
            print("found", zeros, "points with distance of 0")
        return short_dist

    def fit(self, X = None, y = None, test = True):

        def find_outliers(dataset, outlier_constant = 1.5):
            #defintion of outlier w/ 1.5 iqr definition
            upper_quartile = np.percentile(dataset, 75)
            lower_quartile = np.percentile(dataset, 25)
            IQR = (upper_quartile - lower_quartile) * outlier_constant
            outliers = dataset[dataset >= upper_quartile + IQR]
            non_outliers = dataset[dataset < upper_quartile + IQR]
            print(outliers)
            return outliers, non_outliers

        def add_secondary():
            # creating secondary distribution for outliers
            self.secondary_dist = [None for i in range(len(self.labels))] #the secondary distribution is a expo dist
            for label in self.distance.keys():
                #only need to find distance to same class
                distances = np.asarray(self.distance[label][label])
                outliers, non_outliers = find_outliers(distances)
                if len(outliers) == 0: #if no outliers do not add a class
                    pass
                else: #if there are outliers add a new class and remove the outliers
                    self.secondary_dist[label] = np.mean(outliers)
                    # second [lowest_new_class] to make sure other code works
#                     lowest_new_class += 1 # be able to make a new class

        if not X == None == y:
            if len(X) != len(y):
                print("X and y do not have the same length")
                raise NameError
            self.data = X
            self.labels = y


        # store all the distances in format Actual Class: To Class: [closest distances]
        self.distance = defaultdict(lambda: defaultdict(list))


        ### KD tree impementation of distance
        if self.kd:
            self.trees = {}
            #create the KD tree for each class
            for label in set(self.labels):
                #print(self.labels == label)
                self.trees[label] = KDTree(self.data[self.labels == label])

            '''
            Save the distance for each for debug purposes
            '''
            self.debug = defaultdict(dict)
            #from tree find the closest that is not the same point
            # should be able to optize later by sending groups of points
            for i in range(len(self.data)):
                for label, tree in self.trees.items():
                    #set k to 2 so that if same point found, use the second point
                    dist, ind = tree.query([self.data[i]], k = 2)
                    if label == self.labels[i]: #if same class, there will be a point w/ dist 0, sef
                        self.distance[self.labels[i]][label].append(dist[0][1])
                        self.debug[label][i] = dist[0][1]
                    else:
                        self.distance[self.labels[i]][label].append(dist[0][0])

        elif test:
            for i in range(len(self.data)):
                shortests = self.distances(self.data[i])
                for key, shortest in shortests.items():
                    self.distance[self.labels[i]][key].append(shortest)

            # print(self.distance)
#             self.mle()
            # add_secondary()


    def get_details(self):
        return self.distance

    def get_gamma_alphas(self):
        return self.gamma_alphas

    def mle(self, model = "", iterations = 5):

        def gamma_approx():
            #using Gamma(a,beta) not Gamma(alpha, theta)
            alphas = np.zeros((len(set(self.labels)), 2)) # 0 is k, 1 is theta
            x = np.zeros((len(set(self.labels)), 2)) #0 is np.log(np.mean(x)) 1 is np.mean(np.log(x))
            for cat in set(self.labels):
#                 print("Catigory:",self.distance[cat][cat])
                #print(x, self.distance)
                x[cat][0] = np.log(np.mean(self.distance[cat][cat]))
                x[cat][1] = np.mean(np.log(self.distance[cat][cat]))

            alphas[:,0] = .5/(x[:,0] - x[:,1])

            k = alphas[:,0]
            for i in range(iterations):
                digamma = sp.special.digamma(k)
                digamma_prime = sp.special.polygamma(1, k)
                k = 1/ (1/k + (x[:,1] - x[:,0] + np.log(k) - digamma)/(k**2*(1/k - digamma_prime)))
                #print("itermidiary step:", k)

            alphas[:, 0] = k
            alphas[:, 1] = np.exp(x[:, 0])/alphas[:, 0]
            return alphas

        if model == "gamma" or (model == "" and self.model == "gamma"): #[1]
            self.model = "gamma"

            self.gamma_alphas = gamma_approx()
            print("made the alphas")

        else:
            print("Model is not supported")

    def predict(self, data, model = "", explicit = True):
        if model == "gamma" or (model == "" and self.model == "gamma"):

            min_dists = self.distances(data)
            theta = self.gamma_alphas[:,0]
            k = self.gamma_alphas[:,1]
            predictions = np.zeros((self.gamma_alphas.shape[0],1))

            # create the models
            models = [sp.stats.gamma(theta[a], scale = k[a]) for a in min_dists.keys()]

            for cat, dist in min_dists.items():
                predictions[cat] = 1 - models[cat].cdf(dist)
                # get prediction for each class

            """# Uncomment for secondary distribution
                if self.secondary_dist[cat] != None:
                    m = 1/self.secondary_dist[cat]
                    secondary_pred = np.e**(-m * dist) # calc p value for expo funct
                    if secondary_pred > .5:
                        secondary_pred -= .5
                        # should not be too close for the secondary distribution

                    predictions[cat] = max([secondary_pred, predictions[cat]])
            """
            if not explicit:
                prediction = np.argmax(predictions) if predictions[np.argmax(predictions)] >= 1/np.count(self.labels, np.argmax(predictions)) else -1
            return predictions if explicit else prediction

    def score(self, model = "", explicit = False):
        if explicit:
            all_data = []
        if model == "":
            if self.model == "gamma":
                total = 0
                correct = 0
                pred_6 = 0
                for i in range(len(self.data)):
                    predictions = self.predict(self.data[i])
                    if explicit:
                        all_data.append(predictions)
                    else:
                        predict = np.argmax(predictions) if predictions[np.argmax(predictions)] >= self.threshold else -1
                        # print(f"predicted {predict}, should have been {self.labels[i]} because {predictions}")
                        if predict == self.labels[i]:
                            correct += 1
                        else:
                            print(f"distances are {self.distances(self.data[i])}\npredicted class of {predict} when actual was {self.labels[i]}")
                            print(f"the predictions were {predictions}")
                        total += 1
                print(f"the gammas alphas were {self.gamma_alphas}")
                return all_data if explicit else correct/total
