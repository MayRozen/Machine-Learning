import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

class LinearRegressionClassifier:
    def __init__(self):
        self.model = MultiOutputRegressor(LinearRegression())
        self.scaler = StandardScaler()

    def extract_features(self, audio_path):
        """ Extracts audio features: RMS energy, spectral centroid, and max frequency """
        y, sr = librosa.load(audio_path, sr=None)

        # Compute RMS energy (loudness)
        rms = librosa.feature.rms(y=y).flatten()

        # Compute spectral centroid (weighted mean frequency)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()

        # Compute max frequency using spectral rolloff (90% energy threshold)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.90).flatten()

        # Aggregate features: use mean and max for stability
        features = np.array([
            np.mean(rms), np.max(rms),
            np.mean(spectral_centroid), np.max(spectral_centroid),
            np.mean(spectral_rolloff), np.max(spectral_rolloff)
        ])
        return features

    def train(self, X_train, y_train):
        """ Fits a multi-output linear regression model on the training data """
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Make sure y_train has at least two dimensions
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        """ Predicts continuous values for new audio samples """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def evaluate(self, X_test, y_test, categories):
        """ Evaluates the regression model performance for each category """
        y_pred = self.predict(X_test)

        # Debugging prints
        print("DEBUG: y_test shape before reshape:", y_test.shape)
        print("DEBUG: y_pred shape:", y_pred.shape)
        print("DEBUG: Categories:", categories)

        # Ensure y_test is 2D
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
            print("DEBUG: Reshaped y_test to:", y_test.shape)

        # Single-output case
        if y_test.shape[1] == 1:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"[{categories[0]}] MSE: {mse:.4f}, R² Score: {r2:.4f}")
        else:
            for i, category in enumerate(categories):
                mse = mean_squared_error(y_test[:, i], y_pred[:, i])
                r2 = r2_score(y_test[:, i], y_pred[:, i])
                print(f"[{category}] MSE: {mse:.4f}, R² Score: {r2:.4f}")

        self.plot_results(y_test, y_pred, categories)

    def plot_results(self, y_test, y_pred, categories):
        """ Plots true vs predicted values for each category """

        # Ensure y_test is 2D
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        fig, axes = plt.subplots(1, max(1, y_test.shape[1]), figsize=(5 * y_test.shape[1], 5))
        if y_test.shape[1] == 1:
            axes = [axes]  # Ensure indexing works for single-output

        for i, category in enumerate(categories[:y_test.shape[1]]):  # Ensure valid indexing
            axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5, edgecolors='black')
            axes[i].plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], 'r--')
            axes[i].set_title(category)
            axes[i].set_xlabel("True")
            axes[i].set_ylabel("Predicted")

        plt.tight_layout()
        plt.show()
