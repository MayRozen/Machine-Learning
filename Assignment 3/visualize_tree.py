import matplotlib.pyplot as plt

def draw_node(ax, center, text, color):
    """
    Draw a node with the given text and color at the specified position.
    """
    bbox_props = dict(boxstyle="round,pad=0.3", ec="black", fc=color, alpha=0.8)
    ax.text(*center, text, ha="center", va="center", fontsize=10, bbox=bbox_props)

def draw_tree(ax, node, x, y, x_offset, y_offset, level=0):
    """
    Recursively draw the decision tree.
    """
    if node is None:
        return

    # Node text (feature, threshold, or prediction)
    if node.prediction is not None:
        # Map the prediction (1 or 2) to the actual class names
        label_map = {1: "Iris-Versicolor", 2: "Iris-Virginica"}
        prediction_text = label_map.get(node.prediction, f"Unknown ({node.prediction})")
        text = f"Predict\n{prediction_text}"
        color = "#FFCCCB"  # Light Coral
    else:
        feature_names = {0: 'width of sepal', 1: 'length of petal'}
        text = f"{feature_names.get(node.feature, f'Feature {node.feature}')}\n<= {node.threshold}"
        color = "#ADD8E6"  # Light Blue

    # Draw the current node
    draw_node(ax, (x, y), text, color)

    # Draw edges and recursively draw children
    if node.left:
        child_x = x - x_offset / (2 ** level)
        child_y = y - y_offset
        ax.plot([x, child_x], [y, child_y], "k-", lw=1.5)
        draw_tree(ax, node.left, child_x, child_y, x_offset, y_offset, level + 1)

    if node.right:
        child_x = x + x_offset / (2 ** level)
        child_y = y - y_offset
        ax.plot([x, child_x], [y, child_y], "k-", lw=1.5)
        draw_tree(ax, node.right, child_x, child_y, x_offset, y_offset, level + 1)

def visualize_tree_with_matplotlib(tree, filename, title, bg_color):
    """
    Visualize a decision tree and save it as an image.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(bg_color)
    ax.axis("off")

    # Initial parameters for tree drawing
    x = 0.5  # X position of the root
    y = 0.9  # Y position of the root
    x_offset = 0.4  # Horizontal offset between nodes
    y_offset = 0.15  # Vertical offset between levels

    # Draw the tree
    draw_tree(ax, tree, x, y, x_offset, y_offset)

    # Add title with handwriting style
    ax.set_title(title, fontsize=16, fontweight="bold", color="black", fontname="Comic Sans MS")  # Change the font!

    # Save the visualization
    plt.savefig(f"{filename}.png", bbox_inches="tight")
    plt.show()
