# K-means clustering fixed grouping size
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class k_means():

    def __init__(self, K, n_iterations):
        self.K = K # number of clusters
        self.n_iterations = n_iterations
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # get random numbers for clusters
        indices = np.random.choice(X.shape[0], replace=False, size=self.K)
        self.centroids = X[indices].astype(float) # assignments 

        for i in range(self.n_iterations):        
            labels = np.array([np.argmin([self.euclidean_distance(x, self.centroids[k]) for k in range(self.K)]) for x in X])
            self.labels = labels

            for k in range(self.K):
                cluster_points = X[labels == k]
                if len(cluster_points) == 0:
                    continue
                self.centroids[k] = np.mean(cluster_points, axis = 0)

    def predict(self, X):
        return np.array([np.argmin([self.euclidean_distance(x, self.centroids[k]) for k in range(self.K)]) for x in X])

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2))
    

# test on dummy dataset
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

km = k_means(K=4, n_iterations=50)
km.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=km.labels)
plt.scatter(km.centroids[:, 0], km.centroids[:, 1], c='red', marker='x', s=200)
plt.show()
