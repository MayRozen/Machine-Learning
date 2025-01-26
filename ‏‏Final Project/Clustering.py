from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


class AudioClustering:
    def __init__(self, n_clusters=10):
        """
        Initialize Audio Clustering

        Args:
            n_clusters (int): Number of clusters
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def cluster(self, X):
        """
        Perform clustering

        Args:
            X (np.array): Input features

        Returns:
            np.array: Cluster labels
        """
        return self.model.fit_predict(X)

    def evaluate_clustering(self, X):
        """
        Evaluate clustering quality

        Args:
            X (np.array): Input features

        Returns:
            float: Silhouette score
        """
        return silhouette_score(X, self.cluster(X))