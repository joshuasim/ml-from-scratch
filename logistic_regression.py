#logistic regression implementation
import numpy as np 

class logistic_regression():
    # same init arguments as linear regression
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weight = 0
        self.bias = 0

    # sigmoid function to squash z = weight * X + bias between 0 and 1
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    # predict function
    def predict(self, X):
        z = self.weight * X + self.bias
        pred = self.sigmoid(z)
        return pred