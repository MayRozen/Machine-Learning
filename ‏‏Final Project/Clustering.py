import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class ClusteringClassifier:
    def __init__(self, n_clusters=10, max_iter=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.centroid_history = []

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, fmax=sr / 2)
        return np.mean(mfccs, axis=1)

    def preprocess_data(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def initialize_centroids(self, X):
        random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_idx]

    def assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, X, labels):
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def train(self, X):
        self.centroids = self.initialize_centroids(X)
        self.centroid_history.append(self.centroids.copy())

        for i in range(self.max_iter):
            labels = self.assign_labels(X)
            new_centroids = self.update_centroids(X, labels)

            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break

            self.centroids = new_centroids
            self.centroid_history.append(self.centroids.copy())

        self.labels = labels

    def evaluate_clustering(self, X_train, X_test):
        train_labels = self.assign_labels(X_train)
        test_labels = self.assign_labels(X_test)

        unique_clusters = np.unique(test_labels)
        print(f"Unique Clusters Assigned: {unique_clusters}")

        silhouette = silhouette_score(X_test, test_labels)
        print(f"Silhouette Score: {silhouette:.4f}")

        self.visualize_clustering_progress(X_test, test_labels)

        return silhouette

    def visualize_clustering_progress(self, X, labels):
        # List of sound labels corresponding to the clusters
        sound_labels = [
            "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", "Drilling",
            "Engine Idling", "Gun Shot", "Jackhammer", "Siren", "Street Music"
        ]

        # Reduce the data to 2D using PCA for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))

        # Plot the points based on their cluster labels with corresponding colors
        for i in range(self.n_clusters):
            cluster_points = X_2d[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{sound_labels[i]}', s=50, alpha=0.6)

        # Draw circles around each cluster (optional)
        for i in range(self.n_clusters):
            center = self.centroids[i]
            center_2d = pca.transform([center])[0]
            # Draw circles around the clusters, adjusting the size based on your preference
            circle = plt.Circle((center_2d[0], center_2d[1]), 2, color='gray', fill=False, linestyle='--')
            plt.gca().add_artist(circle)

        # Set the title and labels for axes
        plt.title("Clustering Progress: UrbanSound8K")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        # Add the legend to the plot
        plt.legend()

        # Save the figure with a specific filename, overwriting it each time
        plt.savefig('Clustering_Pic.png', bbox_inches='tight')
        plt.show()

