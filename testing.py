import numpy as np
import classifier
import sklearn.model_selection
import matplotlib.pyplot as plt

data= np.genfromtxt('my_datasets/abalone.data', delimiter=",")

#add preprocessing function
x = data[:, 1:-1]
y = data[:, -1]

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y)

model = classifier.NNClassifier()
model.fit(xTrain, yTrain)
print(yTest.shape)
predictions = model.predict(xTest)
print(predictions.shape)

plt.plot(predictions, yTest)
plt.show()
