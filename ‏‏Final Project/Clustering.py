import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class ClusteringClassifier:
    def __init__(self, n_clusters=10, max_iter=100, tolerance=1e-4):
        """
        Initialize the clustering model

        Args:
            n_clusters (int): Number of clusters
            max_iter (int): Maximum number of iterations for the algorithm
            tolerance (float): Convergence tolerance (when centroids don't change significantly)
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def extract_features(self, audio_path):
        """
        Extract MFCC features from an audio file

        Args:
            audio_path (str): Path to the audio file

        Returns:
            np.array: Extracted MFCC features (mean of MFCCs)
        """
        y, sr = librosa.load(audio_path, sr=None)

        # If you are getting empty filters, try adjusting the fmax or n_mels parameters
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, fmax=sr / 2)

        # Alternatively, you can increase the sampling rate if needed
        # y, sr = librosa.load(audio_path, sr=22050)  # Example: setting sr explicitly

        return np.mean(mfccs, axis=1)

    def preprocess_data(self, X):
        """
        Normalize the features using StandardScaler

        Args:
            X (np.array): List of feature vectors

        Returns:
            np.array: Scaled features
        """
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def initialize_centroids(self, X):
        """
        Initialize centroids by selecting random data points

        Args:
            X (np.array): The feature dataset

        Returns:
            np.array: Randomly initialized centroids
        """
        random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_idx]

    def assign_labels(self, X):
        """
        Assign labels to each data point based on the closest centroid

        Args:
            X (np.array): The feature dataset

        Returns:
            np.array: Cluster labels
        """
        # Compute the distance to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, X, labels):
        """
        Recalculate centroids based on the average of assigned points

        Args:
            X (np.array): The feature dataset
            labels (np.array): The cluster labels

        Returns:
            np.array: Updated centroids
        """
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def train(self, X):
        """
        Train the clustering model using the K-means algorithm

        Args:
            X (np.array): The feature dataset
        """
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)

        for i in range(self.max_iter):
            print(f"Iteration {i + 1}...")
            # Step 1: Assign labels
            labels = self.assign_labels(X)

            # Step 2: Update centroids
            new_centroids = self.update_centroids(X, labels)

            # Step 3: Check for convergence (centroids don't change much)
            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                print(f"Converged at iteration {i + 1}")
                break

            # Update centroids for next iteration
            self.centroids = new_centroids

        # Store the final labels
        self.labels = labels

    def evaluate_clustering(self, X_train, X_test):
        """
        Evaluate clustering performance using Silhouette Score

        Args:
            X_train (np.array): The feature dataset used for training
            X_test (np.array): The feature dataset used for evaluation

        Returns:
            float: Silhouette score
        """
        # Assign labels for the training set
        train_labels = self.assign_labels(X_train)

        # Assign labels for the test set based on the centroids learned during training
        test_labels = self.assign_labels(X_test)

        # Evaluate the clustering performance using the test labels
        return silhouette_score(X_test, test_labels)

    def train_and_evaluate(self, audio_paths):
        """
        Train and evaluate the clustering model

        Args:
            audio_paths (list): List of paths to audio files

        Returns:
            float: Silhouette score
        """
        print("Step 1: Extracting features from audio files...")
        features = np.array([self.extract_features(path) for path in audio_paths])

        print(f"Step 2: Preprocessing data, total samples: {features.shape[0]}...")
        features_scaled = self.preprocess_data(features)

        print("Step 3: Training the clustering model...")
        self.train(features_scaled)  # Only pass the features, no labels needed

        print("Training completed, checking silhouette score...")

        print("Step 4: Evaluating clustering performance...")
        silhouette = self.evaluate_clustering(features_scaled,
                                              features_scaled)  # Make sure both X_train and X_test are passed

        print("Step 5: Printing results...")
        print(f"Clustering completed with {self.n_clusters} clusters.")
        print(f"Silhouette Score: {silhouette:.2f}")
        print(f"Cluster labels for each audio file: {self.labels}")

        return silhouette


