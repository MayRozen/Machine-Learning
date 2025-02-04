import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import pairwise_distances
from collections import Counter


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        """
        Initialize KNN Classifier

        Args:
            n_neighbors (int): Number of neighbors to use
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def extract_features(self, audio_path):
        """
        Extract features from an audio file

        Args:
            audio_path (str): Path to the audio file

        Returns:
            np.array: Extracted features (MFCCs)
        """
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)  # Mean of MFCCs as feature vector

    def preprocess_data(self, X, y):
        """
        Normalize and split data into train and test sets

        Args:
            X (np.array): Features
            y (np.array): Labels

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Store the training data and labels

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict using KNN algorithm

        Args:
            X_test (np.array): Test features

        Returns:
            np.array: Predicted labels
        """
        predictions = []
        distances = pairwise_distances(X_test, self.X_train)  # Efficient distance computation

        for i in range(X_test.shape[0]):
            nearest_neighbors = np.argsort(distances[i])[:self.n_neighbors]
            neighbor_labels = self.y_train[nearest_neighbors]

            # Majority voting with tie-breaking
            label_counts = Counter(neighbor_labels)
            predicted_label = max(label_counts.keys(), key=lambda label: (
                label_counts[label], -np.mean(distances[i, nearest_neighbors][neighbor_labels == label])
            ))
            predictions.append(predicted_label)

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Classification Report
        class_report = classification_report(y_test, y_pred)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print results
        print(f"Overall Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(class_report)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\n" + "-" * 50 + "\n")