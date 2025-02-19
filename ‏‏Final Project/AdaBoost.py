import numpy as np
from knn import KNNClassifier  # Import your KNN classifier


class AdaBoost:
    def __init__(self, n_estimators=20, k=3):
        """
        Initialize AdaBoost with KNN classifiers for multi-class classification using SAMME.

        Args:
            n_estimators (int): Number of weak learners.
            k (int): Number of neighbors in the KNN classifier.
        """
        self.n_estimators = n_estimators
        self.k = k
        self.models = []
        self.alphas = []
        self.classes = None  # Unique classes found in training data.
        self.class_to_index = {}  # Mapping from class label to index.

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the AdaBoost model using the SAMME algorithm.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_test (np.array): Optional test features.
            y_test (np.array): Optional test labels.
        """
        N = X_train.shape[0]
        w = np.ones(N) / N  # Initialize uniform sample weights

        # Store unique classes and create a mapping from class label to index.
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError("AdaBoost requires at least 2 classes.")
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

        # Optionally evaluate initial accuracy on test set.
        initial_accuracy = self.evaluate(X_test, y_test) if (X_test is not None and y_test is not None) else None
        if initial_accuracy is not None:
            print(f"Initial accuracy: {initial_accuracy:.4f}")

        for m in range(self.n_estimators):
            # Train a weak learner (KNN in this case) with the current weights.
            model = KNNClassifier(n_neighbors=self.k)
            model.train(X_train, y_train)

            # Predict on the training data.
            y_pred = model.predict(X_train)
            # for debugging if needed -> print(f"Iteration {m + 1}: y_pred: {y_pred}")

            # Compute weighted error: only count misclassified samples.
            incorrect = (y_pred != y_train)
            error = np.dot(w, incorrect)  # Since w sums to 1, this is a weighted error.
            error = max(error, 1e-10)  # Avoid division by zero.

            # Compute alpha using the SAMME formula for multi-class boosting.
            # Note: In binary AdaBoost, alpha = 0.5 * ln((1-error)/error)
            # For multi-class SAMME, we use: alpha = ln((1-error)/error) + ln(n_classes - 1)
            alpha = np.log((1 - error) / error) + np.log(n_classes - 1)
            # Optionally clip alpha to a reasonable range to avoid numerical issues.
            alpha = np.clip(alpha, -2, 2)
            # for debugging if needed -> print(f"Iteration {m + 1}: error = {error}, alpha = {alpha}")

            # Update the weights: increase the weight for misclassified samples.
            w = w * np.exp(alpha * incorrect.astype(np.float64))
            w /= np.sum(w)  # Normalize the weights

            # Save the weak learner and its weight.
            self.models.append(model)
            self.alphas.append(alpha)

        # Optionally evaluate final accuracy on test set.
        if X_test is not None and y_test is not None:
            final_accuracy = self.evaluate(X_test, y_test)
            print(f"Final accuracy after boosting: {final_accuracy:.4f}")
            if initial_accuracy is not None:
                print(f"Improvement in accuracy: {final_accuracy - initial_accuracy:.4f}")

    def predict(self, X_test):
        """
        Predict class labels for the test set.

        Args:
            X_test (np.array): Test features.

        Returns:
            np.array: Predicted class labels.
        """
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        # Initialize a score matrix: rows for samples, columns for classes.
        scores = np.zeros((n_samples, n_classes))

        # Aggregate weighted predictions from each weak learner.
        for model, alpha in zip(self.models, self.alphas):
            y_pred = model.predict(X_test)
            for i, pred in enumerate(y_pred):
                class_index = self.class_to_index[pred]
                scores[i, class_index] += alpha

        # The final prediction is the class with the highest aggregated score.
        final_pred_indices = np.argmax(scores, axis=1)
        final_pred = np.array([self.classes[i] for i in final_pred_indices])
        return final_pred

    def evaluate(self, X_test, y_test):
        """
        Compute the accuracy of the AdaBoost model.

        Args:
            X_test (np.array): Test features.
            y_test (np.array): True labels.

        Returns:
            float: Accuracy score.
        """
        if X_test is None or y_test is None:
            return None
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy
