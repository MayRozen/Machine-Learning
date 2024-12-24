import matplotlib.pyplot as plt


def plot_iris_dataset(x, y):
    """
    Plot the Iris dataset with three species and their numerical labels.

    Params:
        x: Feature matrix (2D NumPy array)
        y: Label vector (1D NumPy array with species as numbers, e.g. 0, 1, 2)
    """
    # Masks for each of the three species
    setosa_mask = y == 0
    versicolor_mask = y == 1
    virginica_mask = y == 2

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Scatter plot for the three classes
    plt.scatter(x[setosa_mask][:, 0], x[setosa_mask][:, 1], c='orange', label='Setosa (0)', edgecolors='k')
    plt.scatter(x[versicolor_mask][:, 0], x[versicolor_mask][:, 1], c='blue', label='Versicolor (1)', edgecolors='k')
    plt.scatter(x[virginica_mask][:, 0], x[virginica_mask][:, 1], c='purple', label='Virginica (2)', edgecolors='k')

    # Add text labels to the points
    for i in range(len(x)):
        plt.text(x[i, 0], x[i, 1], str(y[i]), fontsize=8, color='black')

    # Add labels, title, legend, and grid
    plt.xlabel("Feature 1 (Sepal Width)", fontsize=12)
    plt.ylabel("Feature 2 (Petal Length)", fontsize=12)
    plt.legend(fontsize=10)
    plt.title("Iris Dataset (Three Species)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Show the plot
    plt.show()