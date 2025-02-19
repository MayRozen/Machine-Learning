import numpy as np
from knn import KNNClassifier  # Import your KNN classifier

class AdaBoost:
    def __init__(self, n_estimators=20, k=3):
        """
        Initialize AdaBoost with KNN classifiers.

        Args:
            n_estimators (int): Number of weak models.
            k (int): Number of neighbors in KNN.
        """
        self.n_estimators = n_estimators
        self.k = k
        self.models = []
        self.alphas = []

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train AdaBoost model.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_test (np.array): Optional - test features.
            y_test (np.array): Optional - test labels.
        """
        N = X_train.shape[0]
        w = np.ones(N) / N  # Initialize uniform weights

        # Convert labels to binary {-1, 1}
        unique_classes = np.unique(y_train)
        # Mapping: first unique class -> -1, second -> 1 (make sure your data is binary)
        class_mapping = {c: i * 2 - 1 for i, c in enumerate(unique_classes)}
        y_train = np.vectorize(class_mapping.get)(y_train)

        # Save initial accuracy if test data is provided
        initial_accuracy = self.evaluate(X_test, y_test) if X_test is not None and y_test is not None else None
        if initial_accuracy is not None:
            print(f"Initial accuracy: {initial_accuracy:.4f}")

        for estimator in range(self.n_estimators):
            # Train weak learner (KNN in this case)
            model = KNNClassifier(n_neighbors=self.k)
            model.train(X_train, y_train)

            # Get predictions on training data
            y_pred = model.predict(X_train)
            print(f"Iteration {estimator+1}: y_pred before mapping: {y_pred}")  # Debugging output

            # Convert predictions to binary {-1, 1} using the same mapping
            # Note: Ensure that the KNN always returns values in your expected domain.
            y_pred = np.array([class_mapping.get(y, -1) for y in y_pred])
            print(f"Iteration {estimator+1}: y_pred after mapping: {y_pred}")  # Debugging output

            # Compute weighted error; np.sum(w * (y_pred != y_train)) sums the misclassified weights
            err = np.sum(w * (y_pred != y_train)) / np.sum(w)
            # Prevent error from being too small to avoid division by zero in log calculations
            err = max(err, 1e-5)
            print(f"Iteration {estimator+1}: error = {err}")

            # Compute alpha (the weight of the weak learner) using AdaBoost formula
            alpha = 0.5 * np.log((1 - err) / err)
            # Clip alpha to the range [-2, 2] to avoid extreme values that might cause numerical issues
            alpha = np.clip(alpha, -2, 2)
            print(f"Iteration {estimator+1}: alpha = {alpha}")

            # Update weights:
            # Instead of directly multiplying by exp(-alpha * y_train * y_pred), we use logarithms to help prevent underflow/overflow
            w = np.exp(np.log(w) - alpha * y_train * y_pred)
            # Normalize the weights so that they sum to 1
            w /= np.sum(w)

            # Store the weak learner and its corresponding alpha
            self.models.append(model)
            self.alphas.append(alpha)

        # Final accuracy after boosting
        if X_test is not None and y_test is not None:
            final_accuracy = self.evaluate(X_test, y_test)
            print(f"Final accuracy after boosting: {final_accuracy:.4f}")
            if initial_accuracy is not None:
                print(f"Improvement in accuracy: {final_accuracy - initial_accuracy:.4f}")

    def predict(self, X_test):
        """
        Predict using the trained AdaBoost model.

        Args:
            X_test (np.array): Test features.

        Returns:
            np.array: Predicted labels.
        """
        # Initialize prediction sum for each test example
        y_pred_sum = np.zeros(X_test.shape[0])

        for model, alpha in zip(self.models, self.alphas):
            # Get prediction from each weak learner
            y_pred = model.predict(X_test)
            # Map prediction: if y_pred is 0, map to -1, else 1
            # (Make sure this mapping is consistent with your label representation)
            y_pred_mapped = np.where(y_pred == 0, -1, 1)
            # Aggregate the weighted predictions
            y_pred_sum += alpha * np.sign(y_pred_mapped)

        # Return the sign of the sum as the final prediction
        return np.sign(y_pred_sum)

    def evaluate(self, X_test, y_test):
        """
        Compute model accuracy.

        Args:
            X_test (np.array): Test features.
            y_test (np.array): True labels.

        Returns:
            float: Accuracy score.
        """
        if X_test is None or y_test is None:
            return None

        # Convert true labels to {-1, 1} using the same mapping as in training
        unique_classes = np.unique(y_test)
        class_mapping = {c: i * 2 - 1 for i, c in enumerate(unique_classes)}
        y_test = np.vectorize(class_mapping.get)(y_test)

        # Get predictions from the model
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy
