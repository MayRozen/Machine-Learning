import numpy as np
from itertools import product
from collections import Counter
from visualize_tree import visualize_tree_with_matplotlib


# Helper functions
def entropy(labels):
    """
    Compute binary entropy.
    """
    n = len(labels)
    if n == 0:
        return 0
    probabilities = [count / n for count in Counter(labels).values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


def split_data(data, labels, feature, threshold):
    """
    Split data and labels based on a feature and threshold.
    """
    left_mask = data[:, feature] <= threshold
    right_mask = ~left_mask
    return (data[left_mask], labels[left_mask]), (data[right_mask], labels[right_mask])


def error_rate(labels):
    """
    Compute error rate assuming the majority class prediction.
    """
    if len(labels) == 0:
        return 0
    majority_label = Counter(labels).most_common(1)[0][0]
    errors = sum(1 for label in labels if label != majority_label)
    return errors / len(labels)


def brute_force_decision_tree(data, labels, max_depth):
    """
    Decision Tree using brute-force to find the optimal solution.
    Creates a tree structure using `Node` instances.
    """
    n_features = data.shape[1]  # Number of features

    def build_tree(data, labels, depth):
        # Stop if there is no data, max depth reached, or all labels are the same
        if len(data) == 0 or depth == max_depth or len(set(labels)) == 1:
            prediction = Counter(labels).most_common(1)[0][0]  # Majority prediction
            return Node(prediction=prediction)

        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None
        lowest_error = float("inf")  # Start with a very large error to minimize

        for feature in range(n_features):
            thresholds = np.unique(data[:, feature])  # All unique values for the split
            for threshold in thresholds:
                # Split data into left and right based on the threshold
                (left_data, left_labels), (right_data, right_labels) = split_data(data, labels, feature, threshold)
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue  # Skip invalid splits (division that results in empty groups)

                # Calculate error for the current split
                left_error = error_rate(left_labels)
                right_error = error_rate(right_labels)
                weighted_error = (len(left_labels) * left_error + len(right_labels) * right_error) / len(labels)

                # Update the best split
                if weighted_error < lowest_error:
                    lowest_error = weighted_error
                    best_feature = feature
                    best_threshold = threshold
                    best_left = (left_data, left_labels)
                    best_right = (right_data, right_labels)

        # If no split found, return a leaf node with majority prediction
        if best_feature is None:
            prediction = Counter(labels).most_common(1)[0][0]
            return Node(prediction=prediction)

        # Create a new node and recursively build subtrees
        root = Node()
        root.feature = best_feature
        root.threshold = best_threshold
        root.left = build_tree(*best_left, depth + 1)
        root.right = build_tree(*best_right, depth + 1)
        return root

    # Build the tree starting from depth 0
    tree = build_tree(data, labels, 0)

    # Calculate the error rate for the built tree
    def calculate_tree_error(tree, data, labels):
        """
        Helper function to calculate the error rate of a decision tree.
        """

        def predict(node, sample):
            if node.prediction is not None:  # If it’s a leaf node, return the prediction
                return node.prediction
            if sample[node.feature] <= node.threshold:
                return predict(node.left, sample)
            return predict(node.right, sample)

        predictions = [predict(tree, sample) for sample in data]
        errors = sum(1 for true_label, pred_label in zip(labels, predictions) if true_label != pred_label)
        return errors / len(labels)

    brute_error = calculate_tree_error(tree, data, labels)

    return tree, brute_error



# Binary Entropy Decision Tree
class Node:
    def __init__(self, prediction=None):
        self.prediction = prediction
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


def binary_entropy_decision_tree(data, labels, max_depth, depth=0):
    """
    Construct a decision tree using binary entropy minimization.
    """
    if depth == max_depth or len(set(labels)) == 1:
        return Node(prediction=Counter(labels).most_common(1)[0][0])

    n_features = data.shape[1]
    thresholds = np.unique(data.flatten())
    best_split = None
    lowest_entropy = float("inf")

    # Find the best split
    for feature in range(n_features):
        for threshold in thresholds:
            (left_data, left_labels), (right_data, right_labels) = split_data(data, labels, feature, threshold)
            left_entropy = entropy(left_labels)
            right_entropy = entropy(right_labels)
            weighted_entropy = (len(left_labels) * left_entropy + len(right_labels) * right_entropy) / len(labels)
            if weighted_entropy < lowest_entropy:
                lowest_entropy = weighted_entropy
                best_split = (feature, threshold, left_data, left_labels, right_data, right_labels)

    if best_split is None:
        return Node(prediction=Counter(labels).most_common(1)[0][0])

    feature, threshold, left_data, left_labels, right_data, right_labels = best_split
    root = Node()
    root.feature = feature
    root.threshold = threshold
    root.left = binary_entropy_decision_tree(left_data, left_labels, max_depth, depth + 1)
    root.right = binary_entropy_decision_tree(right_data, right_labels, max_depth, depth + 1)
    return root

def decision_tree_main():
    """
    Main function to run and compare the decision tree algorithms
    using brute force and binary entropy methods on the Iris dataset.
    """
    # Import necessary library for dataset
    from sklearn.datasets import load_iris

    # Load the Iris dataset
    iris = load_iris()

    # Filter the data for Versicolor (label 1) and Virginica (label 2)
    data = iris.data
    labels = iris.target
    filtered_indices = (labels == 1) | (labels == 2)  # Keep only labels 1 and 2
    data = data[filtered_indices]  # Filter data
    labels = labels[filtered_indices]  # Filter labels

    # Use only the second and third features, as required
    data = data[:, 1:3]

    # Define the maximum depth for the decision trees (k=3, as specified)
    max_depth = 3

    # Run the brute-force decision tree algorithm
    print("Running Brute-Force Decision Tree...")
    tree1, brute_error = brute_force_decision_tree(data, labels, max_depth)
    print("Brute Force - Best Tree:", tree1)
    print("Brute Force - Error Rate:", brute_error)

    # Run the binary entropy decision tree algorithm
    print("\nRunning Binary Entropy Decision Tree...")
    tree2 = binary_entropy_decision_tree(data, labels, max_depth)

    # Debug tree structures
    print("\nTree Structures:")

    def debug_tree_structure(node, depth=0):
        if node is None:
            print(f"{'  ' * depth}Empty node")
            return
        print(f"{'  ' * depth}Node: Feature={node.feature}, Threshold={node.threshold}, Prediction={node.prediction}")
        if node.left:
            debug_tree_structure(node.left, depth + 1)
        if node.right:
            debug_tree_structure(node.right, depth + 1)

    print("Tree 1 (Brute Force):")
    debug_tree_structure(tree1)
    print("\nTree 2 (Binary Entropy):")
    debug_tree_structure(tree2)

    # Calculate error rate for Binary Entropy Decision Tree
    def calculate_tree_error(tree, data, labels):
        """
        Helper function to calculate error rate of a decision tree.
        """

        def predict(node, sample):
            if node.prediction is not None:
                return node.prediction
            if sample[node.feature] <= node.threshold:
                return predict(node.left, sample)
            return predict(node.right, sample)

        predictions = [predict(tree, sample) for sample in data]
        errors = sum(1 for true_label, pred_label in zip(labels, predictions) if true_label != pred_label)
        return errors / len(labels)

    entropy_error = calculate_tree_error(tree2, data, labels)
    print(f"Binary Entropy Decision Tree Error Rate: {entropy_error:.4f}")

    # Visualize each tree and save them
    print("\nVisualizing Decision Trees...")
    visualize_tree_with_matplotlib(tree1, "brute_force_tree", "Brute Force Decision Tree", "#FFDAB9")  # Peach
    visualize_tree_with_matplotlib(tree2, "binary_entropy_tree", "Binary Entropy Decision Tree",
                                   "#B0E0E6")  # Light Blue

    # Summary and comparison of results
    print("\nSummary:")
    print(f"Brute-Force Decision Tree Error Rate: {brute_error}")
    print(f"Binary Entropy Decision Tree Error Rate: {entropy_error:.4f}")