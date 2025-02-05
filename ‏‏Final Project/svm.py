import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', max_iter=1000, gamma=0.1):
        """
        Implementing SVM using SMO (Sequential Minimal Optimization)
        C: Regularization parameter
        kernel: Type of kernel ('linear' or 'rbf')
        max_iter: Maximum number of iterations for training
        gamma: Kernel parameter for RBF kernel
        """
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.y_support = None

    def linear_kernel(self, X1, X2):
        """
        Linear kernel function
        """
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2):
        """
        Efficient RBF (Radial Basis Function) kernel computation
        Computes the kernel matrix between two sets of samples.
        """
        X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]  # Squared norms of X1
        X2_sq = np.sum(X2**2, axis=1)  # Squared norms of X2
        squared_dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)  # Squared Euclidean distance
        return np.exp(-self.gamma * squared_dist)  # Apply RBF kernel formula

    def kernel(self, X1, X2):
        """
        Select kernel type: RBF or Linear
        """
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            return self.linear_kernel(X1, X2)

    def train(self, X_train, y_train):
        """
        Training the SVM algorithm using the training data
        """
        n_samples, n_features = X_train.shape
        self.alpha = np.zeros(n_samples)  # Lagrange multipliers
        self.y_support = np.where(y_train == 0, -1, 1)  # Convert {0,1} labels to {-1,1} for SVM

        # Precompute kernel matrix for efficiency
        K = self.kernel(X_train, X_train)

        # Precompute constant dot product between alpha * y_support and kernel matrix for efficiency
        alpha_y_support = self.alpha * self.y_support
        kernel_dot_product = np.dot(alpha_y_support, K) - self.b - self.y_support

        # Main SMO optimization loop
        for _ in range(self.max_iter):
            for i in range(n_samples):
                E_i = kernel_dot_product[i]  # Calculate error for sample i

                # Conditions for updating alpha_i
                if (self.y_support[i] * E_i < -1e-3 and self.alpha[i] < self.C) or (
                        self.y_support[i] * E_i > 1e-3 and self.alpha[i] > 0):

                    # Select j intelligently based on the largest error
                    j = np.argmax(np.abs(E_i - kernel_dot_product))

                    if i == j:
                        continue

                    E_j = kernel_dot_product[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    # Compute L and H, bounds for alpha values
                    if self.y_support[i] != self.y_support[j]:
                        L, H = max(0, alpha_j_old - alpha_i_old), min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L, H = max(0, alpha_i_old + alpha_j_old - self.C), min(self.C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    # Calculate eta, the denominator for alpha update
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha[j]
                    self.alpha[j] -= (self.y_support[j] * (E_i - E_j)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)  # Ensure alpha[j] is within bounds

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha[i]
                    self.alpha[i] += self.y_support[i] * self.y_support[j] * (alpha_j_old - self.alpha[j])

                    # Recompute kernel_dot_product with new alpha values
                    kernel_dot_product = np.dot(self.alpha * self.y_support, K) - self.b - self.y_support

                    # Update bias term b
                    b1 = self.b - E_i - self.y_support[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - self.y_support[
                        j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y_support[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - self.y_support[
                        j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

        # Save support vectors after training
        self.support_vectors = X_train[self.alpha > 0]
        self.y_support = self.y_support[self.alpha > 0]
        self.alpha = self.alpha[self.alpha > 0]

        print(f"Training completed. Support vectors count: {len(self.support_vectors)}")

    def predict(self, X_test):
        """
        Predict labels for the test set
        """
        K = self.kernel(X_test, self.support_vectors)  # Compute kernel matrix with support vectors
        predictions = np.sign(np.dot(K, self.alpha * self.y_support) - self.b)  # Decision function
        return predictions

    def extract_features(self, audio_path):
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features from an audio file
        This will be used as input for the SVM model
        """
        y, sr = librosa.load(audio_path, sr=None)  # Load audio file
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features (13 coefficients)
        return np.mean(mfccs, axis=1)  # Return mean of the MFCCs as feature vector

    def preprocess_data(self, X, y):
        """
        Normalize features and split data into training and test sets
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 split
        scaler = StandardScaler()  # Standardize data to have zero mean and unit variance
        X_train = scaler.fit_transform(X_train)  # Fit scaler to training data
        X_test = scaler.transform(X_test)  # Transform test data using the same scaler
        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on the test data
        """
        predictions = self.predict(X_test)  # Get predictions from the model
        accuracy = accuracy_score(y_test, predictions)  # Calculate accuracy
        print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")  # Print accuracy in percentage
        return accuracy

# Hyperparameter tuning using GridSearchCV
def tune_hyperparameters(X_train, y_train):
    """
    Tune hyperparameters (C and gamma) using GridSearchCV for better performance
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf']  # You can also include 'linear' kernel if you want to test both
    }
    grid_search = GridSearchCV(SVMClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_
