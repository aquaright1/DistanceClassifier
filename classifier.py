### Classifier ###
from sklearn.preprocessing import LabelEncoder
from helpers import closest_linear, gamma_mle
from collections import defaultdict
import scipy as sp

class NNClassifier():
    def __init__(self, ε = 0.0001: float, threshold = .01: float):
        self.ε = ε
        self.threshold = threshold

    def fit(self,X,y):
        # save the inputs into the model
        # also make sure it is a numpy array to make things easier
        self.input_data = np.asarray(X)


        # use label encoder to transform labels into integers
        # from 0->n where n is the number of different classes
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        self.encoded_labels = self.encoder.transform(y)

        # initiate the parameters and the minimum distances
        self.params = np.zeros((len(self.encoder.classes_), 3))
        self.distances = defaultdict(list)

        # search for the minimum distance to points of the same class
        for index, data in enumerate(self.input_data):
            label = self.encoded_labels
            distances[label].append(closest_linear(data,self.input_data[self.encoded_labels == label], fit = True))

        for key in self.distances.keys():
            # take the log of the minium distances
            self.distances[key] = np.log(np.asarray(self.distances[key]))

            # move data points into support of gamma distribution and save the movement
            minimum = np.min(self.distances[key])
            self.params[key][2] = minimum
            self.distances[key] -= minimum
            self.distances[key] += self.ε

            # find the parameters for gamma distribution and save them
            self.params[key][0], self.params[key][1] = gamma_mle(self.distances[key])

    def predict(self,X, p-value = False):
        '''
        X: data points for classification
        full: whether or not to give all the p-scores

        returns: np array of predictions or np array of nparrays of p-scores
        '''

        # put in a place for models to be able to saved
        models = [i for i in set(self.encoded_labels)]

        # each data point we want to predict on needs an array that is of the same length as the
        # number of classes they can be classified into
        predictions = [models.copy() for x in X]

        # for each class test each point
        for class in set(self.encoded_labels):
            # save the models
            models[class] = sp.stats.gamma(self.params[class][0], self.params[class][1])
            for index, x in enumerate(X):
                # calculate distances and shift into gamma's support
                dist = closest_linear(x, self.input_data[self.encoded_labels == class])
                adj_dist = dist - self.params[class][2] + ε

                # calculate the inverse cdf
                # if the adjusted isn't above zero, it is closer to a member of said class than
                # any other point
                predictions[index][class] = 1-models[class].cdf(adj_dist) if adj_dist > 0 else 1

        predictions = np.asarray(predictions)
        if p-value:
            return predictions
        prediction = np.argmax(predictions, axis = 1)
        '''
        TODO: parallized this following code:
        '''
        for index, individual in enumerate(predictions):
            if individual[prediction[index]] < self.threshold:
                prediction[index] = -1


        return prediction








    #only here so that i can scroll down while coding
