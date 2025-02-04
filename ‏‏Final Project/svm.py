import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', max_iter=1000, gamma=0.1):
        """
        Implementing SVM using SMO
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
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2):
        """ Efficient RBF kernel computation """
        X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
        X2_sq = np.sum(X2**2, axis=1)
        squared_dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * squared_dist)

    def kernel(self, X1, X2):
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            return self.linear_kernel(X1, X2)

    def fit(self, X_train, y_train):
        """
        Training the algorithm on voice data
        """
        n_samples, n_features = X_train.shape
        self.alpha = np.zeros(n_samples)
        self.y_support = np.where(y_train == 0, -1, 1)  # Convert {0,1} to {-1,1}

        K = self.kernel(X_train, X_train)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                E_i = np.dot(self.alpha * self.y_support, K[:, i]) - self.b - self.y_support[i]
                if (self.y_support[i] * E_i < -1e-3 and self.alpha[i] < self.C) or (
                        self.y_support[i] * E_i > 1e-3 and self.alpha[i] > 0):

                    # Select j intelligently
                    j = np.argmax(np.abs(E_i - (np.dot(self.alpha * self.y_support, K) - self.b - self.y_support)))
                    if i == j:
                        continue
                    E_j = np.dot(self.alpha * self.y_support, K[:, j]) - self.b - self.y_support[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    # Compute L and H
                    if self.y_support[i] != self.y_support[j]:
                        L, H = max(0, alpha_j_old - alpha_i_old), min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L, H = max(0, alpha_i_old + alpha_j_old - self.C), min(self.C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha[j]
                    self.alpha[j] -= (self.y_support[j] * (E_i - E_j)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha[i]
                    self.alpha[i] += self.y_support[i] * self.y_support[j] * (alpha_j_old - self.alpha[j])

                    # Compute b
                    b1 = self.b - E_i - self.y_support[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - self.y_support[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y_support[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - self.y_support[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

        # Save support vectors
        self.support_vectors = X_train[self.alpha > 0]
        self.y_support = self.y_support[self.alpha > 0]
        self.alpha = self.alpha[self.alpha > 0]

    def predict(self, X_test):
        """
        Predicting labels for new audio clips
        """
        K = self.kernel(X_test, self.support_vectors)
        return np.sign(np.dot(K, self.alpha * self.y_support) - self.b)

    def extract_features(self, audio_path):
        """
        Extracting MFCC features from an audio file
        """
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)

    def preprocess_data(self, X, y):
        """
        Normalization and division into training set and test set
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
