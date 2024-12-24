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

    # Scatter plot for each species with distinct modern colors
    ax.scatter(x[setosa_mask][:, 0], x[setosa_mask][:, 1],
               color='#FF7518', label='Setosa (0)', edgecolors='black', s=120, alpha=0.8)
    ax.scatter(x[versicolor_mask][:, 0], x[versicolor_mask][:, 1],
               color='#0096FF', label='Versicolor (1)', edgecolors='black', s=120, alpha=0.8)
    ax.scatter(x[virginica_mask][:, 0], x[virginica_mask][:, 1],
               color='#6B5B95', label='Virginica (2)', edgecolors='black', s=120, alpha=0.8)

    # Customize the axes labels
    ax.set_xlabel("Feature 1 (Sepal Width)", fontsize=16, fontweight='bold', color='#333333')
    ax.set_ylabel("Feature 2 (Petal Length)", fontsize=16, fontweight='bold', color='#333333')

    # Set the title with styling
    ax.set_title("Iris Dataset - Professional Visualization", fontsize=18, fontweight='bold', color='#333333', pad=15)

    # Tweak the appearance of the grid and ticks
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12, color='gray', labelcolor='#333333')

    # Add a sleek legend with custom font size and transparency
    legend = ax.legend(fontsize=14, frameon=True, loc='upper left', fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.8)

    # Add panel background and customize its color
    ax.set_facecolor('#F5F5F5')  # Light gray background for the panel

    # Display the plot
    plt.tight_layout()  # Adjust layout to avoid overlaps
    plt.show()