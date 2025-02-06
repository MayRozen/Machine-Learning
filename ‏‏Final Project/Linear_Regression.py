import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionClassifier:
    def __init__(self):
        """Initialize Linear Regression model"""
        self.weights = None
        self.bias = None
        self.X_train = None
        self.y_train = None

    def extract_features(self, audio_path):
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.array: Extracted features (mean of MFCCs)
        """
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC coefficients
        return np.mean(mfccs, axis=1)  # Take the mean of the MFCCs over time

    def preprocess_data(self, X, y):
        """
        Preprocess the data by splitting it into training and test sets, and standardizing the features.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Labels vector.

        Returns:
            tuple: Scaled training and test sets (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # Standardize training data
        X_test = scaler.transform(X_test)  # Standardize test data using the same parameters
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the Linear Regression model using Normal Equation.

        Args:
            X_train (np.array): Training feature matrix.
            y_train (np.array): Training target vector.
        """
        self.X_train = X_train
        self.y_train = y_train

        # Add bias term to the features (X_train)
        X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]

        # Compute the weights using the Normal Equation
        # weights = (X^T * X)^-1 * X^T * y
        self.weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        self.bias = self.weights[0]  # bias term is the first element of weights
        self.weights = self.weights[1:]  # remaining weights are the coefficients

    def predict(self, X_test):
        """
        Predict the target values using the trained Linear Regression model.

        Args:
            X_test (np.array): Test feature matrix.

        Returns:
            np.array: Predicted values
        """
        X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # Add bias term to the features
        return X_test.dot(np.r_[self.bias, self.weights])  # Linear prediction

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R²).

        Args:
            X_test (np.array): Test feature matrix.
            y_test (np.array): True target vector.
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")
        print("\n" + "-" * 50 + "\n")

        self.plot_results(y_test, y_pred)

    def plot_results(self, y_test, y_pred):
        """
        Plot the results of the predictions.

        Args:
            y_test (np.array): True values.
            y_pred (np.array): Predicted values.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', edgecolors='black', label='Predictions')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
                 label='Perfect Prediction')
        plt.title("Linear Regression Results", fontsize=16, fontweight="bold", fontname="Arial")
        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.legend(loc="upper left", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
