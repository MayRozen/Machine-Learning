import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionDB:
    def __init__(self):
        """ Initialize the model and scalers """
        self.scaler = StandardScaler()  # Standard scaler for features
        self.y_scaler = StandardScaler()  # Standard scaler for target values (dB)
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def extract_features(self, audio_path):
        """ Extract key features from an audio file: Root Mean Square (RMS) energy """
        y, sr = librosa.load(audio_path, sr=None)

        # Compute RMS energy (average loudness)
        rms = librosa.feature.rms(y=y).flatten()

        # Use the mean RMS value as a stable metric
        mean_rms = np.mean(rms)
        return np.array([mean_rms]).reshape(1, -1), y, sr, rms  # Return extra values for visualization

    def train(self, X_train, y_train):
        """ Train the model on loudness levels """
        X_train_scaled = self.scaler.fit_transform(X_train)  # Scale input features
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1))  # Scale target values

        # Add a bias column to the feature matrix (for the intercept term)
        X_train_scaled_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]

        # Calculate weights using the Normal Equation: (X^T X)^-1 X^T y
        X_transpose = X_train_scaled_bias.T
        self.w = np.linalg.inv(X_transpose.dot(X_train_scaled_bias)).dot(X_transpose).dot(y_train_scaled)

        # The first element of w corresponds to the bias term (intercept)
        self.b = self.w[0]
        self.w = self.w[1:]  # Remove the bias term from w

    def predict(self, X_test):
        """ Predict the target values from input features """
        X_test_scaled = self.scaler.transform(X_test)  # Scale the test data

        # Add a column of ones to represent the bias term (intercept)
        X_test_scaled_bias = np.c_[
            np.ones(X_test_scaled.shape[0]), X_test_scaled]  # Add bias (intercept) as the first column

        # Concatenate bias and weights, ensuring correct dimensionality for dot product
        weights_with_bias = np.concatenate(([self.b], self.w))  # Combine bias and weights in the correct order

        # Calculate predictions (dot product with weights and bias)
        return X_test_scaled_bias.dot(weights_with_bias)  # Dot product to calculate predictions

    def evaluate(self, X_test, y_test):
        """ Evaluate model performance using MSE and R^2 score """
        predicted_values = self.predict(X_test)
        predicted_values = self.y_scaler.inverse_transform(predicted_values.reshape(-1, 1))  # Inverse transform

        # Calculate and print evaluation metrics
        mse = mean_squared_error(y_test, predicted_values)
        r2 = r2_score(y_test, predicted_values)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

        self.check_noise_regulation(predicted_values, day=True)
        return mse, r2

    def check_noise_regulation(self, predicted_dB, day=True):
        """ Check if the predicted noise level exceeds legal limits in Israel
            Daytime: 55-65 dB
            Nighttime: 45-55 dB
        """
        legal_limit = (55, 65) if day else (45, 55)

        # If predicted_dB is an array, iterate over each value
        if isinstance(predicted_dB, np.ndarray):
            for predicted in predicted_dB:
                self._check_single_noise_level(predicted, legal_limit)
        else:
            # Handle case where predicted_dB is a single value
            self._check_single_noise_level(predicted_dB, legal_limit)

    def _check_single_noise_level(self, predicted, legal_limit):
        """ Helper function to check a single noise level """
        # Ensure predicted is a scalar (e.g., float), not an array
        predicted = predicted.item() if isinstance(predicted, np.ndarray) else predicted

        if predicted < legal_limit[0]:
            print(f"Noise level is acceptable ({predicted:.2f} dB).")
        elif predicted > legal_limit[1]:
            print(f"Noise level EXCEEDS the legal limit! ({predicted:.2f} dB)")
        else:
            print(f"Noise level is within the legal range ({predicted:.2f} dB).")
