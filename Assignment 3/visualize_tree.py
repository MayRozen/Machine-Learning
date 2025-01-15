import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os


def visualize_tree_with_matplotlib(tree, filename, title, color):
    """
    Visualize a decision tree using matplotlib.

    Parameters:
    tree (Node): The root of the decision tree.
    filename (str): The filename for saving the visualization (without extension).
    title (str): The title for the tree chart.
    color (str): Background color of tree nodes.
    """

    fig, ax = plt.subplots(figsize=(14, 8))  # Adjusted size for better visualization
    ax.axis("off")  # Turn off axis lines

    # Recursive function to plot nodes and their connections
    def plot_node(node, x, y, dx, level):
        if node is None:
            return

        # Create the node label to show its content
        if node.prediction is not None:  # Leaf node
            label = f"Predict: {node.prediction}"
        else:  # Decision node
            label = f"Feature {node.feature}\nThreshold {node.threshold:.2f}"

        # Draw the node with text
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.5'))

        # Plot children if exist
        next_y = y - 1.5  # Reduce y for the next level
        if node.left:  # Left child node
            ax.plot([x, x - dx], [y - 0.2, next_y + 0.2], lw=1, color='black')  # Connection line
            plot_node(node.left, x - dx, next_y, dx / 2, level + 1)

        if node.right:  # Right child node
            ax.plot([x, x + dx], [y - 0.2, next_y + 0.2], lw=1, color='black')  # Connection line
            plot_node(node.right, x + dx, next_y, dx / 2, level + 1)

    # Start the plotting from the root node
    plot_node(tree, x=0, y=0, dx=8, level=0)

    # Add title to the plot
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Save the plot to a file
    save_path = os.getcwd()
    plt.savefig(f"{save_path}/{filename}.png", bbox_inches="tight")
    print(f"Tree visualization saved as {filename}.png at {save_path}")

    # Show the plot
    plt.show()