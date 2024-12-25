# Python 3.11.9

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import random  # Import random for random sampling


# Helper function to define a line's decision rule
def line_rule(x, point1, point2):
    """Return 1 if x is above the line, otherwise -1."""
    x1, y1 = point1
    x2, y2 = point2
    # Define the slope and y-intercept
    slope = (y2 - y1) / (x2 - x1 + 1e-10)  # Add small epsilon to avoid division by zero
    intercept = y1 - slope * x1
    return lambda data: np.sign(data[:, 1] - (slope * data[:, 0] + intercept))


# Load the Iris dataset and filter by classes Versicolor and Virginica
iris = load_iris()
X = iris.data[iris.target != 0, :2]  # Use only the first two features
y = iris.target[iris.target != 0]
y = np.where(y == 1, 1, -1)  # Versicolor: 1, Virginica: -1

# Total iterations for Adaboost
NUM_RUNS = 100
NUM_HYPOTHESES = 8

# Store errors for Train and Test
train_errors = np.zeros(NUM_HYPOTHESES)
test_errors = np.zeros(NUM_HYPOTHESES)

# Run Adaboost 100 times
for run in range(NUM_RUNS):
    # Split data randomly into 50% train (S) and 50% test (T)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    # Generate hypothesis set H using random pairs of points
    num_random_hypotheses = 50  # Number of random hypotheses to generate
    hypotheses = []
    for _ in range(num_random_hypotheses):
        # Randomly select two indices from the training data
        idx1, idx2 = random.sample(range(len(X_train)), 2)
        point1, point2 = X_train[idx1], X_train[idx2]
        # Add a classifier based on the random line between the two points
        hypotheses.append(line_rule(X_train, point1, point2))

    # Initialize weights
    n_train = len(y_train)
    weights = np.ones(n_train) / n_train
    alpha = []
    selected_hypotheses = []

    # Adaboost iterations
    for t in range(NUM_HYPOTHESES):
        # Find the weak classifier with the lowest weighted error
        errors = []
        for h in hypotheses:
            preds = h(X_train)
            errors.append(np.sum(weights * (preds != y_train)))
        errors = np.array(errors)
        best_h_index = np.argmin(errors)
        best_h = hypotheses[best_h_index]
        selected_hypotheses.append(best_h)

        # Calculate alpha
        epsilon = errors[best_h_index]  # Error of best hypothesis
        alpha_t = 0.5 * np.log((1 - epsilon) / (epsilon + 1e-10))
        alpha.append(alpha_t)

        # Update weights
        preds = best_h(X_train)
        weights *= np.exp(-alpha_t * y_train * preds)
        weights /= np.sum(weights)  # Normalize

    # Compute errors for each k=1,...,8
    H_train = np.zeros(len(y_train))
    H_test = np.zeros(len(y_test))
    for k in range(NUM_HYPOTHESES):
        h_k = selected_hypotheses[k]
        a_k = alpha[k]
        H_train += a_k * h_k(X_train)
        H_test += a_k * h_k(X_test)

        # Evaluate using final combined hypotheses
        H_train_preds = np.sign(H_train)
        H_test_preds = np.sign(H_test)
        train_errors[k] += np.mean(H_train_preds != y_train)
        test_errors[k] += np.mean(H_test_preds != y_test)

# Average errors over the 100 runs
train_errors /= NUM_RUNS
test_errors /= NUM_RUNS

# Print results
for k in range(NUM_HYPOTHESES):
    print(f"\nH{k + 1}: Train Error = {train_errors[k]:.4f}, Test Error = {test_errors[k]:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_HYPOTHESES + 1), train_errors, label="Train Error", marker='o', color='#5ea9d6')
plt.plot(range(1, NUM_HYPOTHESES + 1), test_errors, label="Test Error", marker='x', color='#ff8a9f')

# Define axis labels and title with a handwriting font


plt.xlabel("Number of Hypotheses (k)", fontsize=14, fontname="Comic Sans MS")
plt.ylabel("Error Rate", fontsize=14, fontname="Comic Sans MS")
plt.title("Adaboost Performance on Train and Test Sets", fontsize=18, fontname="Comic Sans MS")

# Stylize the remaining plot
plt.legend()
plt.grid()
plt.show()