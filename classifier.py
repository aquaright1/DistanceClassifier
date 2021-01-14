### Classifier ###
from sklearn.preprocessing import LabelEncoder
from helpers import closest_linear, shift, gamma_mle
from collections import defaultdict

class NNClassifier():
    def __init__(self, ε = 0.0001: float, threshold = .01: float):
        self.ε = ε
        self.threshold = threshold

    def fit(self,X,y):
        '''
        TODO:
            implment the fit
        '''
        self.input_data = np.asarray(X)
        self.encoder = LabelEncoder()
        self.encoder.fit(y)

        self.encoded_labels = self.encoder.transform(y)

        self.params = np.zeros((len(self.encoder.classes_), 3))

        self.distances = defaultdict(list)
        for index, data in enumerate(self.input_data):
            label = self.encoded_labels
            distances[label].append(closest_linear(data,self.input_data[self.encoded_labels == label], fit = True))

        for key in self.distances.keys():
            self.distances[key] = np.log(np.asarray(self.distances[key]))
            minimum = np.min(self.distances[key])
            self.params[key][2] = minimum
            self.distances[key] -= minimum
            self.distances[key] += self.ε
            self.params[key][0], self.params[key][1] = gamma_mle(self.distances[key])
            
    def predict(self,X):
        '''
        TODO:
            implement the prediciton
        '''
        pass
