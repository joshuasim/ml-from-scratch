#linear regression
import numpy as np

class linear_regression():
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weight = 0
        self.bias = 0

    def predict(self, X):
        y = self.weight * X + self.bias
        return y

    def fit(self, X, y):
        for i in range(self.n_iterations):
            y_pred = self.predict(X)
            res = y_pred - y
            #gradient calculation for weight and nias
            w_grad = np.mean(res * 2 * X)
            b_grad = np.mean(res * 2)
            #update
            self.weight = self.weight - (self.learning_rate * w_grad)
            self.bias = self.bias - (self.learning_rate* b_grad)

# 1. predict 2. calc loss function* 3. chain rule 4. update w and b 5. repeat n_iterations

#dummy dataset
X = np.linspace(0,10,100)
y = 3*X+5
regression = linear_regression(1000, .01)
regression.fit(X,y)
print(regression.weight, regression.bias)

from sklearn.linear_model import LinearRegression
sk_model = LinearRegression()
sk_model.fit(X.reshape(-1,1), y)
print(sk_model.coef_, sk_model.intercept_)