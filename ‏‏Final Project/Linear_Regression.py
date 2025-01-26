from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class AudioLinearRegression:
    def __init__(self):
        """Initialize Linear Regression model"""
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """
        Train linear regression model

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict using trained model

        Args:
            X_test (np.array): Test features

        Returns:
            np.array: Predicted values
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test (np.array): Test features
            y_test (np.array): True values

        Returns:
            dict: Regression metrics
        """
        y_pred = self.predict(X_test)
        return {
            'mean_squared_error': mean_squared_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }