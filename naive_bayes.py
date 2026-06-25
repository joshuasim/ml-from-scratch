# naive bayes
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB 

# a classification algorithm based on bayes theorem that assumes features are independent of one another and therefore is naive. 
class naive_bayes():
    def __init__(self):
        self.prior_prob = {}
        self.mean = {}
        self.std = {}
        self.classes = None
    
    # fit sets the classes and gets all the means, std, and prior_probs for all the classes. 
    def fit(self, X, y):
        self.classes = np.unique(y) # set classes as a np array of unique y

        # iterate through the classes in self.classes and calculate the prior_prob, mean, and std for each class, using a dictionary to store the class and value.
        for classs in self.classes:
            self.prior_prob[classs] = np.sum(y==classs) / len(y)
            self.mean[classs] = np.mean(X[y==classs], axis=0)
            self.std[classs] = np.std(X[y==classs], axis=0)

    # helper function to calculate the gaussian probability distribution formula, and then returns the log of that prob to avoid underflow
    def gaussian_pdf(self, x, mean, std):
        prob = (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2/(2*std**2))
        log_prob = np.log(prob)
        return log_prob

    # predicts which class has the highest probability
    def predict(self, x):
        # track the highest prob and its class
        highest = float('-inf') # init at -inf so that any log prob will be greater
        highest_class = None 
        # calculate prob for each class and then compare with the overall highest prob, if the prob is greater than the current highest -> redefine variables
        for c in self.classes:  
            prob = np.log(self.prior_prob[c]) + np.sum(self.gaussian_pdf(x, self.mean[c], self.std[c]))
            if prob > highest:
                highest = prob
                highest_class = c 

        return highest_class
        

# dummy dataset to test algorithm
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)

# call my implementation 
bayes = naive_bayes()
bayes.fit(X, y)
# since predict function only handles a single sample, so i use a for loop to iterate through X and then convert the list of predictions into a np array
pred = []
for i in X:
    pred.append(bayes.predict(i))
pred = np.array(pred)
accuracy = np.mean(pred == y)
print(f'scratch model accuracy: {accuracy}')
# 0.905

# call sklearn bayes implementation
sklearn_bayes = GaussianNB()
sklearn_bayes.fit(X, y)
pred_sklearn = sklearn_bayes.predict(X)
sklearn_bayes_accuracy = np.mean(pred_sklearn == y)
print(f'sklearn model accuracy: {sklearn_bayes_accuracy}')
# 0.905