#logistic regression implementation
import numpy as np 

class logistic_regression():
    # same init arguments as linear regression
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weight = 0 # weight is initalized at 0 here just to show it exists. If left as is during fit it will cause error because of matrix multiplication.
        self.bias = 0

    # sigmoid function to squash z = weight * X + bias between 0 and 1
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    # predict function
    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
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
        self.weight = np.zeros(X.shape[1]) # this ensures that matrix multiplication can be done
        for i in range(self.n_iterations):
            y_pred = self.predict(X)
            N = X.shape[0]
            grad_w = (1/N) * np.dot(X.T, (y_pred - y))           
            grad_b = np.mean(y_pred - y) # grad_b = (1/N) * (y_pred - y)
            self.weight = self.weight - (self.learning_rate * grad_w)
            self.bias = self.bias - (self.learning_rate * grad_b)

# test with dummy dataset

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)

reg = logistic_regression(1000, 0.01)
reg.fit(X, y)
pred = reg.predict_class(X)
accuracy = np.mean(pred == y)
print(f'scratch model accuracy: {accuracy}')
# 0.9

# sklearn
sklearn_reg = LogisticRegression()
sklearn_reg.fit(X, y)
pred_sklearn = sklearn_reg.predict(X)
skleanr_reg_accuracy = np.mean(pred_sklearn == y)
print(f'sklearn model accuracy: {skleanr_reg_accuracy}')
#0.903

# my implementation and sklearn's performed almost equally well, 0.9 and 0.903 respectively,
# but sklearn performed 0.3% better than my implementation. this difference is due to sklearn using 
# a more advanced optimization algorithm (lbfgs) compared to vanilla gradient descent.

