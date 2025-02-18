import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', max_iter=1000, gamma=0.1):
        """
        Implementing SVM using SMO (Sequential Minimal Optimization)
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
        X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
        X2_sq = np.sum(X2**2, axis=1)
        squared_dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * squared_dist)

    def kernel(self, X1, X2):
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            return self.linear_kernel(X1, X2)

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.alpha = np.zeros(n_samples)
        self.y_support = np.where(y_train == 0, -1, 1)
        K = self.kernel(X_train, X_train)
        alpha_y_support = self.alpha * self.y_support
        kernel_dot_product = np.dot(alpha_y_support, K) - self.b - self.y_support

        for _ in range(self.max_iter):
            for i in range(n_samples):
                E_i = kernel_dot_product[i]
                if (self.y_support[i] * E_i < -1e-3 and self.alpha[i] < self.C) or (
                        self.y_support[i] * E_i > 1e-3 and self.alpha[i] > 0):
                    j = np.argmax(np.abs(E_i - kernel_dot_product))
                    if i == j:
                        continue

                    E_j = kernel_dot_product[j]
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    if self.y_support[i] != self.y_support[j]:
                        L, H = max(0, alpha_j_old - alpha_i_old), min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L, H = max(0, alpha_i_old + alpha_j_old - self.C), min(self.C, alpha_i_old + alpha_j_old)
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= (self.y_support[j] * (E_i - E_j)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += self.y_support[i] * self.y_support[j] * (alpha_j_old - self.alpha[j])
                    kernel_dot_product = np.dot(self.alpha * self.y_support, K) - self.b - self.y_support

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

        self.support_vectors = X_train[self.alpha > 0]
        self.y_support = self.y_support[self.alpha > 0]
        self.alpha = self.alpha[self.alpha > 0]

        print(f"Training completed. Support vectors count: {len(self.support_vectors)}")

    def predict(self, X_test):
        K = self.kernel(X_test, self.support_vectors)
        predictions = np.sign(np.dot(K, self.alpha * self.y_support) - self.b)
        return predictions

    def preprocess_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
        tune_and_visualize_svm(X_test, y_test)
        return accuracy

# Hyperparameter tuning and visualization
def tune_and_visualize_svm(X, y):
    X_train, X_test, y_train, y_test = SVMClassifier().preprocess_data(X, y)

    svm = SVMClassifier(C=1.0, kernel='rbf', gamma=0.5)
    svm.train(X_train, y_train)

    # Check if data has more than 2 features
    if X.shape[1] > 2:
        print("Applying PCA to reduce to 2D for visualization...")
        pca = PCA(n_components=2)
        X_2D = pca.fit_transform(X)
        support_vectors_2D = pca.transform(svm.support_vectors)
    else:
        X_2D = X
        support_vectors_2D = svm.support_vectors

    # Create grid for plotting decision boundary
    xx, yy = np.meshgrid(np.linspace(X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1, 100),
                         np.linspace(X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1, 100))

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Debug: check dimensions
    print(f"Support vectors shape: {support_vectors_2D.shape}")

    # Compute decision function for each point in the grid
    Z = svm.kernel(grid_points, support_vectors_2D)

    # Debug: check dimensions of Z
    print(f"Z shape before dot product: {Z.shape}")

    Z = np.sign(np.dot(Z, svm.alpha * svm.y_support) - svm.b)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.75, colors=['r', 'k', 'g'], linestyles=['--', '-', '--'])

    # Plot data points
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, s=50, cmap='autumn', edgecolors='k')

    # Plot support vectors
    plt.scatter(support_vectors_2D[:, 0], support_vectors_2D[:, 1], s=100, facecolors='none', edgecolors='b',
                linewidths=1.5)

    plt.title("SVM with RBF Kernel: Decision Boundary and Support Vectors", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Save the figure (overwrite existing one)
    plt.savefig("svm_decision_boundary.png", dpi=300)
    print("Plot saved as 'svm_decision_boundary.png'")

    plt.show()
