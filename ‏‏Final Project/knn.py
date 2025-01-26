from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        """
        Initialize KNN Classifier

        Args:
            n_neighbors (int): Number of neighbors to use
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        """
        Train KNN model

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using trained model

        Args:
            X_test (np.array): Test features

        Returns:
            np.array: Predicted labels
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test (np.array): Test features
            y_test (np.array): True labels

        Returns:
            dict: Classification metrics
        """
        y_pred = self.predict(X_test)
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }