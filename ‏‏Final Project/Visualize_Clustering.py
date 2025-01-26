import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA


class ClusteringVisualization:
    def __init__(self, X, cluster_labels):
        """
        Initialize Clustering Visualization

        Args:
            X (np.array): Feature matrix
            cluster_labels (np.array): Assigned cluster labels
        """
        self.X = X
        self.cluster_labels = cluster_labels
        self.pastel_colors = sns.color_palette("pastel")

    def plot_clusters(self):
        """
        Visualize clusters using PCA for dimensionality reduction
        """
        # Reduce dimensionality
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(self.X)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=self.cluster_labels,
            cmap=plt.cm.Pastel1,
            alpha=0.7
        )
        plt.title('Sound Clusters Visualization', fontsize=15)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig('clustering_visualization.png')
        plt.close()

    def plot_silhouette_scores(self, silhouette_scores):
        """
        Plot silhouette scores for different cluster counts

        Args:
            silhouette_scores (list): Silhouette scores for various cluster counts
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
        plt.title('Silhouette Score for Different Cluster Counts', fontsize=15)
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.tight_layout()
        plt.savefig('silhouette_scores.png')
        plt.close()