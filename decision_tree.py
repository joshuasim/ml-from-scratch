# 1 stump decision tree
import numpy as np

class decision_tree():
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.L_leaf = None
        self.R_leaf = None

    def fit():
        def gini(self, y):
            prob_1 = np.sum(y)/y.shape[0]
            prob_0 = 1 - prob_1
            impurity = 1 - (prob_0**2 + prob_1**2)
            return impurity