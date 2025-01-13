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


# Brute-Force Decision Tree
def brute_force_decision_tree(data, labels, max_depth):
    """
    Decision Tree using brute force to find the optimal solution,
    optimized for better performance.
    """
    n_features = data.shape[1]  # Number of features
    thresholds = [np.mean(data[:, feature]) for feature in range(n_features)]  # Use mean for thresholds
    best_tree = None
    lowest_error = float("inf")  # Initialize the lowest error

    def compute_error_and_split(data, labels, feature, threshold):
        """
        Compute error and perform a split based on the feature and threshold.
        Returns the weighted error and left/right splits.
        """
        left_mask = data[:, feature] <= threshold
        right_mask = ~left_mask
        left_labels = labels[left_mask]
        right_labels = labels[right_mask]
        left_error = error_rate(left_labels)
        right_error = error_rate(right_labels)
        # Weighted error based on the size of the splits
        weighted_error = (len(left_labels) * left_error + len(right_labels) * right_error) / len(labels)
        return weighted_error, (data[left_mask], left_labels), (data[right_mask], right_labels)

    # Generate all combinations of features for splits up to max depth
    for feature_combination in product(range(n_features), repeat=max_depth):
        nodes = [(data, labels)]  # Start with the root node
        total_error = 0

        for depth, feature in enumerate(feature_combination):
            current_threshold = thresholds[feature]  # Use the precomputed threshold for this feature
            new_nodes = []
            level_error = 0

            for node_data, node_labels in nodes:
                if len(set(node_labels)) <= 1:  # Stop splitting pure nodes
                    new_nodes.append((node_data, node_labels))
                    continue

                # Perform split and compute error
                split_error, left, right = compute_error_and_split(node_data, node_labels, feature, current_threshold)
                level_error += split_error
                new_nodes.append(left)
                new_nodes.append(right)

            # Update total error for this level and move to the next
            total_error += level_error
            nodes = new_nodes

        # Update the best tree if the current error is lower than the lowest error
        if total_error < lowest_error:
            lowest_error = total_error
            best_tree = feature_combination

    return best_tree, lowest_error


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


def run_decision_tree():
    # Import required library for dataset
    from sklearn.datasets import load_iris
    iris = load_iris()

    # Extract data and labels
    data = iris.data
    labels = iris.target

    # Only select Versicolor (1) and Virginica (2)
    filtered_indices = (labels == 1) | (labels == 2)
    data = data[filtered_indices]
    labels = labels[filtered_indices]

    # Select only the second and third features (coordinates)
    data = data[:, 1:3]

    # Define max depth for both decision trees
    max_depth = 3

    # Run brute force decision tree
    brute_tree, brute_error = brute_force_decision_tree(data, labels, max_depth)
    print("Brute Force - Best Tree:", brute_tree)
    print("Brute Force - Error:", brute_error)

    # Run binary entropy decision tree
    entropy_tree = binary_entropy_decision_tree(data, labels, max_depth)
    print("Binary Entropy - Decision Tree Built")

    # Visualize Decision Tree using matplotlib
    try:
        visualize_tree_with_matplotlib(entropy_tree, filename="entropy_tree")
        print("Binary Entropy Tree - visualization completed")
    except Exception as e:
        print("Error visualizing tree using matplotlib.")
        print(e)
