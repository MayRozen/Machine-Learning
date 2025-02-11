import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionClassifier:
    def __init__(self, alpha=0.01, l2_lambda=0.1, use_gradient_descent=False, max_iter=1000):
        """
        Initialize Linear Regression model.

        Args:
            alpha (float): Learning rate for gradient descent.
            l2_lambda (float): Regularization strength for Ridge Regression.
            use_gradient_descent (bool): Whether to use gradient descent instead of the Normal Equation.
            max_iter (int): Maximum iterations for gradient descent.
        """
        self.weights = None
        self.scaler = StandardScaler()
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.use_gradient_descent = use_gradient_descent
        self.max_iter = max_iter

    def extract_features(self, audio_path):
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.array: Extracted features (mean of MFCCs)
        """
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)

    def train(self, X_train, y_train):
        """
        Train the Linear Regression model using either Normal Equation or Gradient Descent.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]  # Add bias term

        if self.use_gradient_descent:
            self.weights = np.zeros(X_train_scaled.shape[1])
            for i in range(self.max_iter):
                gradients = (X_train_scaled.T @ (X_train_scaled @ self.weights - y_train)) / len(y_train)
                reg_term = self.l2_lambda * self.weights  # Ridge regularization (L2)
                self.weights -= self.alpha * (gradients + reg_term)
        else:
            I = np.eye(X_train_scaled.shape[1])  # Identity matrix for L2 regularization
            I[0, 0] = 0  # Don't regularize bias term
            self.weights = np.linalg.pinv(
                X_train_scaled.T @ X_train_scaled + self.l2_lambda * I) @ X_train_scaled.T @ y_train

        print(f"X_train shape: {X_train_scaled.shape}, weights shape: {self.weights.shape}")

    def predict(self, X_test, scaled=False):
        """
        Predict the target values using the trained Linear Regression model.
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")

        if not scaled:
            X_test = self.scaler.transform(X_test)

        X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        return X_test @ self.weights

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance using MSE and R².
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")
        self.plot_results(y_test, y_pred)

    def plot_results(self, y_test, y_pred):
        """
        Plot the results of the predictions.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', edgecolors='black', label='Predictions')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
                 label='Perfect Prediction')
        plt.title("Linear Regression Results", fontsize=16, fontweight="bold")
        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()