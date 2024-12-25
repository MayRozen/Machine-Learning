import numpy as np

from iris_to_csv import iris_to_csv
from load_iris import load_iris
from the_graph import plot_iris_dataset


class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=10000):
        self.learning_rate = learning_rate  # Learning rate determines step size for weight updates
        self.max_iter = max_iter            # Maximum number of iterations for training
        self.w = None                       # Weight vector
        self.b = None                       # Bias term
        self.mistake_count = None  # Number of mistakes made during training

    def fit(self, x, y):
        """
        Train the Perceptron algorithm.
        :param x: Feature matrix of shape (n_samples, n_features)
        :param y: Label vector of shape (n_samples,) with values {-1, +1}
        """
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)  # Initialize weights to zero
        self.b = 0  # Initialize bias to zero
        self.mistake_count = 0  # Number of mistakes made during training

        for t in range(self.max_iter):  # For round t=1,2,… (max_iter as stopping criterion)
            no_mistakes = True  # Assume no mistakes initially

            for i in range(n_samples):  # Iterate over all points xi
                # If w_t · x_i > 0, guess +
                prediction = np.dot(self.w, x[i]) + self.b
                if prediction > 0:
                    guess = +1
                else:
                    guess = -1

                # On mistake
                if guess != y[i]:
                    self.mistake_count += 1
                    # If x is really +, w_(t+1) ← w_t + x_i
                    if y[i] == +1:
                        self.w += self.learning_rate * x[i]
                    # If x is really -, w_(t+1) ← w_t − x_i
                    else:  # y[i] == -1
                        self.w -= self.learning_rate * x[i]

                    # Update the bias
                    self.b += self.learning_rate * y[i]

                    no_mistakes = False  # Step 9: Exit round t after a mistake
                    break

            # If no mistakes in this round, exit algorithm
            if no_mistakes:
                break

    def predict(self, x):
        """
        Predict labels for new examples.
        :param x: Feature matrix of shape (n_samples, n_features)
        :return: Label vector of shape (n_samples,) with values {-1, +1}
        """
        return np.sign(np.dot(x, self.w) + self.b)  # Compute sign of (w · x + b)

# Main function to run the Perceptron on the Iris dataset
def main():
    iris_to_csv()
    # Load the data (change the file name if necessary)
    x, y = load_iris()

    # Example: To classify only Setosa vs Versicolor
    mask = (y == 0) | (y == 1)  # Filter Setosa (0) and Versicolor (1)
    x_binary = x[mask]
    y_binary = y[mask]
    y_binary = np.where(y_binary == 0, -1, 1)  # Convert Setosa to -1, Versicolor to +1 for binary classification

    # Train your perceptron on the filtered dataset
    perceptron = Perceptron(learning_rate=0.01, max_iter=10000)
    perceptron.fit(x_binary, y_binary)
    predictions = perceptron.predict(x_binary)
    print("predictions: " , predictions)

    # Access the trained weights and bias
    w1 = perceptron.w  # Weight vector
    b1 = perceptron.b  # Bias term
    mistake_count1 = perceptron.mistake_count

    # Print results for Setosa vs Versicolor
    print("Final vector for Setosa vs Versicolor:")
    print(f"Weight vector: {w1}, Bias: {b1}")
    print(f"Total mistakes made: {mistake_count1}")

    max_margin = 1 / np.linalg.norm(w1)
    print(f"Maximum Margin: {max_margin}")

    # Call the plot function to visualize the dataset
    plot_iris_dataset(x, y)

if __name__ == "__main__":
    main()