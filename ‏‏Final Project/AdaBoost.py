from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class AdaBoostAudioClassifier:
    def __init__(self, n_estimators=50):
        """
        Initialize AdaBoost Classifier

        Args:
            n_estimators (int): Number of weak learners
        """
        self.model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators,
            random_state=42
        )

    def train(self, X_train, y_train):
        """
        Train AdaBoost model

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