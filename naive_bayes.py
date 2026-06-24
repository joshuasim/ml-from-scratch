# naive bayes
import numpy as np

class naive_bayes():
    def __init__(self):
        self.prior_prob = {}
        self.mean = {}
        self.std = {}
        self.classes = None
    
    # fit sets the classes and gets all the means, std, and prior_probs for all the classes. 
    def fit(self, X, y):
        self.classes = np.unique(y) # set classes as a np array of unique y

        #iterate through the classes in self.classes and calculate the prior_prob, mean, and std for each class, using a dictionary to store the class and value.
        for classs in self.classes:
            self.prior_prob[classs] = np.sum(y==classs) / len(y)
            self.mean[classs] = np.mean(X[y==classs], axis=0)
            self.std[classs] = np.std(X[y==classs], axis=0)

    # helper function to calculate the guassian probability distribution formula, and then returns the log of that prob to avoid underflow
    def guassian_pdf(self, x, mean, std):
        prob = (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2/(2*std**2))
        log_prob = np.log(prob)
        return log_prob

