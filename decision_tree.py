# 1 stump decision tree
import numpy as np

class decision_tree():
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.L_leaf = None
        self.R_leaf = None

    # iterates through features and thresholds to set the feature index and threshold for the depth-1 tree stump and calculates the left and right leaf.
    def fit(self, x, y):
        best_gini = float('inf') # ensures any weight_gini is less than the initial best_gini
        for feature_idx in range(x.shape[1]):
            thresholds = np.unique(x[:, feature_idx])
            for i in thresholds:
                left_y = y[x[:, feature_idx] <= i]
                right_y = y[x[:, feature_idx] > i]
                gini_left = self.gini(left_y)
                gini_right = self.gini(right_y)
                weighted_gini = (left_y.shape[0]/y.shape[0]) * gini_left + (right_y.shape[0]/y.shape[0])*gini_right
                if weighted_gini < best_gini: # changes best gini and sets the left leaf, right leaf, feature and threshold only when new calculated weighted gini is less than previous best gini
                    best_gini = weighted_gini
                    if int(np.sum(left_y) / len(left_y) >= 0.5): # assigns leaf predictions based on majority class in each split group.
                        self.L_leaf = 1
                        self.R_leaf = 0
                    else:
                        self.R_leaf = 1
                        self.L_leaf = 0
                    self.feature = feature_idx
                    self.threshold = i        

    # gini method to calcuate gini impurity
    def gini(self, y):
        if len(y) == 0: # this ensures no RuntimeWarning for invalid value encontered for division with 0
            return 0
        prob_1 = np.sum(y)/y.shape[0]
        prob_0 = 1 - prob_1
        impurity = 1 - (prob_0**2 + prob_1**2)
        return impurity
    
    # returns predicted class label for each sample based on the learned feature and threshold.
    def predict(self, x):
        return np.where(x[:, self.feature] <= self.threshold, self.L_leaf, self.R_leaf)

# dummy dataset to evaluate. 
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier 
x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)

# call my implementation 
tree = decision_tree()
tree.fit(x, y)
pred = tree.predict(x)
accuracy = np.mean(pred == y)
print(f'scratch model accuracy: {accuracy}')
# 0.903

# call sklearn tree implementation at depth 1
sklearn_tree = DecisionTreeClassifier(max_depth=1)
sklearn_tree.fit(x, y)
pred_sklearn = sklearn_tree.predict(x)
sklearn_tree_accuracy = np.mean(pred_sklearn == y)
print(f'sklearn model accuracy: {sklearn_tree_accuracy}')
# 0.903

#the algorithm is initialized by setting the right and left leaf to none, setting up the feature and 
# threshold also at none. then for the fit you need to calcualte the gini for the different features 
# to know where to split and what value the splits will take. for the actual fit function its just 
# like iterating over the featuers adn the different threshholds which is just the unique values for 
# the features. then u use those thresholds and calcualte their gini_weight and stuff and see if the 
# current threshold has a lower gini than some other higher one if so then you recalculate if not you
# move on then for predict it just sees if the left leaf has more than 50% 1 then itll predict 1 else
# itll be 0/ this is my depth 1 tree summary how is it