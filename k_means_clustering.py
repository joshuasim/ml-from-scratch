# K-means clustering fixed grouping size
import numpy as np

class k_means():

    def __init__(self, K, n_iterations):
        self.K = K # number of clusters
        self.n_iterations = n_iterations
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # get random numbers for clusters
        indices = np.random.choice(X.shape[0], replace=False, size=self.K)
        self.centroids = X[indices] # assignments 

        for i in range(self.n_iterations):        
            labels = np.array([np.argmin([self.euclidean_distance(x, self.centroids[k]) for k in range(len(self.centroids))]) for x in X])
            self.labels = labels
            for k in range(self.K):
                cluster_points = X[labels == k]
                centroid_k = np.mean(cluster_points, axis = 0)
                self.centroids[k] = centroid_k
        
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2))
