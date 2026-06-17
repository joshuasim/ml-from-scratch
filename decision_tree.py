# 1 stump decision tree
import numpy as np

class decision_tree():
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.L_leaf = None
        self.R_leaf = None

    def fit(self, x, y):
        best_gini = float('inf')
        for feature_idx in range(x.shape[1]):
            thresholds = np.unique(x[:, feature_idx])
            for i in thresholds:
                left_y = y[x[:, feature_idx] <= i]
                right_y = y[x[:, feature_idx] > i]
                gini_left = self.gini(left_y)
                gini_right = self.gini(right_y)
                weighted_gini = (left_y.shape[0]/y.shape[0]) * gini_left + (right_y.shape[0]/y.shape[0])*gini_right
                if weighted_gini < best_gini:
                best_gini = weighted_gini
                self.feature = x[feature_idx]
                self.threshold = i
                self.L_leaf = idk
                self.R_leaf = idk

        def gini(self, y):
            prob_1 = np.sum(y)/y.shape[0]
            prob_0 = 1 - prob_1
            impurity = 1 - (prob_0**2 + prob_1**2)
            return impurity