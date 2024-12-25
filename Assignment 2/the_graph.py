import matplotlib.pyplot as plt


def plot_iris_dataset(x, y):
    """
    Professionally styled visualization of the Iris Dataset.

    Params:
        x: Feature matrix (2D NumPy array)
        y: Label vector (1D NumPy array with species as numbers, e.g. 0, 1, 2)
    """
    # Create masks for each species
    setosa_mask = y == 0
    versicolor_mask = y == 1
    virginica_mask = y == 2

    # Use a modern style for the plot
    plt.style.use('seaborn-v0_8-muted')  # Set the style to seaborn muted (minimal and professional)

    # Create a new figure with appropriate dimensions
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot for each species with distinct modern colors (remove edgecolors)
    ax.scatter(x[setosa_mask][:, 0], x[setosa_mask][:, 1],
               color='#FF7518', label='Setosa', s=120, alpha=0.8)
    ax.scatter(x[versicolor_mask][:, 0], x[versicolor_mask][:, 1],
               color='#87CEEB', label='Versicolor', s=120, alpha=0.8)
    ax.scatter(x[virginica_mask][:, 0], x[virginica_mask][:, 1],
               color='#9370DB', label='Virginica', s=120, alpha=0.8)

    # Set axes properties and limits
    ax.set_xlim([0, 9])  # X-axis starts at 0 and ends at 9
    ax.set_ylim([0, 9])  # Y-axis starts at 0 and ends at 9
    ax.set_xticks(range(0, 10, 1))  # Steps of 1 for X-axis
    ax.set_yticks(range(0, 10, 1))  # Steps of 1 for Y-axis

    # Customize the axes labels
    ax.set_xlabel("Feature 1 (Sepal Width)", fontsize=16, fontweight='bold', color='#333333')
    ax.set_ylabel("Feature 2 (Petal Length)", fontsize=16, fontweight='bold', color='#333333')

    # Set the title with styling
    ax.set_title("Iris Dataset - Machine Learning Ex2", fontsize=18, fontweight='bold', color='#333333', pad=15)

    # Tweak the appearance of the grid and ticks
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12, color='gray', labelcolor='#333333')

    # Add panel background and customize its color
    ax.set_facecolor('#FFFFFF')  # White background for the panel

    # Custom Legend: Symbols and text below the graph
    # Handles: Custom circles to match plot
    setosa_circle = plt.Line2D([], [], color='#FF7518', marker='o', linestyle='', markersize=10, label='Setosa')
    versicolor_circle = plt.Line2D([], [], color='#87CEEB', marker='o', linestyle='', markersize=10, label='Versicolor')
    virginica_circle = plt.Line2D([], [], color='#9370DB', marker='o', linestyle='', markersize=10, label='Virginica')

    # Add the legend below the graph
    ax.legend(handles=[setosa_circle, versicolor_circle, virginica_circle],
              loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3, fontsize=12)

    # Display the plot
    plt.tight_layout()  # Adjust layout to avoid overlaps
    plt.show()