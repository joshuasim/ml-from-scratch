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

    # since logistic reg is a classifier this function converts probabilities -> class labels
    def predict_class(self, X):
        return (self.predict(X) > 0.5).astype(int)
    
    
    # Loss: L = (-1/N) * sum(y*log(ŷ) + (1-y)*log(1-ŷ))
    # Gradient: dL/dw = (1/N) * (X^T(ŷ - y))
    # for logistic regression the loss function is log likelihood, so the gradient is the derivative of log loss
    # to calculate this gradient: 
    # 1. compute the diff between predicted probabilities and actual labels
    # 2. multiply by X transpose, X transpose so that the num of rows and cols are equal
    # 3. take mean over N samples to normalize. 
    def fit(self, X, y):
        for i in range(self.n_iterations):
            y_pred = self.predict(X)
            N = X.shape[0]
            grad_w = (1/N) * np.dot(X.T, (y_pred - y))           
            grad_b = np.mean(y_pred - y) # grad_b = (1/N) * (y_pred - y)
            self.weight = self.weight - (self.learning_rate * grad_w)
            self.bias = self.bias - (self.learning_rate * grad_b)