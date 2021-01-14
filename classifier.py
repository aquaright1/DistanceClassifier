### Classifier ###
from sklearn.preprocessing import LabelEncoder
from helpers import closest_linear, gamma_mle
from collections import defaultdict

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

    def predict(self,X, full = False):
        '''
        X: data points for classificationm
        full: whether or not to give all the p-scores

        returns: np array of predictions or np array of nparrays of p-scores
        '''
        pass
