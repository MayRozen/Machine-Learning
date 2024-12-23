import numpy as np

from iris_to_csv import iris_to_csv
from load_iris import load_iris


class Perceptron:
    def __init__(self, learning_rate=1.0, max_iter=1000):
        self.learning_rate = learning_rate  # Learning rate determines step size for weight updates
        self.max_iter = max_iter            # Maximum number of iterations for training
        self.w = None                       # Weight vector
        self.b = None                       # Bias term

    def fit(self, x, y):
        """
        Train the Perceptron algorithm.
        :param x: Feature matrix of shape (n_samples, n_features)
        :param y: Label vector of shape (n_samples,) with values {-1, +1}
        """
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)  # Initialize weights to zero
        self.b = 0  # Initialize bias to zero

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

    # Instantiate the Perceptron
    perceptron = Perceptron(learning_rate=0.1, max_iter=1000)

    # Train the Perceptron
    perceptron.fit(x, y)
    predictions = perceptron.predict(x)
    print(predictions)

    # Access the trained weights and bias
    w1 = perceptron.w  # Weight vector
    b1 = perceptron.b  # Bias term

    # Print results for Setosa vs Versicolor
    print("Final vector for Setosa vs Versicolor:")
    print(f"Weight vector: {w1}, Bias: {b1}")
    # print(f"Number of mistakes: {mistakes1}")

if __name__ == "__main__":
    main()