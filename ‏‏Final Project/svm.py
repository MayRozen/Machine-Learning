import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', max_iter=1000, gamma=0.1, class_weight=None):
        """
        Implementing SVM using SMO (Sequential Minimal Optimization)
        """
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.y_support = None

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
        X2_sq = np.sum(X2 ** 2, axis=1)
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


# Hyperparameter tuning and visualization with GridSearchCVdef tune_and_visualize_svm(X, y):
def tune_and_visualize_svm(X, y):
    # Data distribution and processing – make sure that the data entered is from the full set, not just a test.
    X_train, X_test, y_train, y_test = SVMClassifier().preprocess_data(X, y)

    # Reducing dimensions to 2 for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Setting search parameters
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 0.5, 1.0],
        'kernel': ['rbf'],
        'class_weight': [None, 'balanced']
    }

    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=0, n_jobs=-1)
    grid_search.fit(X_train_pca, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    best_svm = grid_search.best_estimator_

    # Evaluating the best model
    predictions = best_svm.predict(X_test_pca)
    accuracy = np.mean(predictions == y_test)
    print(f"Best SVM Model Accuracy: {accuracy * 100:.2f}%")

    # Creating a grid of points to draw decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # In the case of multi-class classification, we will use predict
    Z = best_svm.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Visualization with contourf for colors based on labels
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # Displaying training points
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=50, cmap='autumn', edgecolors='k')

    # Highlighting support vectors
    support_vectors = best_svm.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='b', linewidths=1.5)

    plt.title("SVM with RBF Kernel: Best Model Decision Boundary", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("best_svm_decision_boundary.png", dpi=300)
    print("Plot saved as 'best_svm_decision_boundary.png'")
    plt.show()
