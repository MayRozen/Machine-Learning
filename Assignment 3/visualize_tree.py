# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyArrowPatch
#
#
# class Node:
#     def __init__(self, prediction=None):
#         self.prediction = prediction
#         self.feature = None
#         self.threshold = None
#         self.left = None
#         self.right = None
#
#
# def visualize_tree_with_matplotlib(root, filename="tree_visualization"):
#     """
#     Visualize a decision tree using matplotlib.
#     This function displays the tree structure by plotting it as a visual diagram.
#
#     Parameters:
#     root (Node): The root node of the decision tree.
#     filename (str): Name of the file to save the visualization (without extension).
#                     If not specified, the tree is displayed using plt.show().
#     """
#     if not root:
#         print("Tree is empty. No visualization generated.")
#         return
#
#     # Initialize figure and axis
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.axis("off")  # Turn off axis visibility
#
#     def plot_node(ax, node, x, y, dx, parent_x=None, parent_y=None):
#         """
#         Recursively plot nodes and connect them with arrows.
#
#         Parameters:
#         ax (matplotlib axes): Axes object to draw on.
#         node (Node): The current node being plotted.
#         x, y (float): Current coordinates for the node position.
#         dx (float): Horizontal distance between nodes.
#         parent_x, parent_y (float): Parent node's coordinates for connecting edges.
#         """
#         if not node:
#             return
#
#         # Display the node information
#         if node.prediction is not None:
#             label = f"Predict: {node.prediction}"
#         else:
#             label = f"Feature {node.feature}\nThreshold {node.threshold:.2f}"
#
#         # Draw the text box to represent the node
#         ax.text(x, y, label, fontsize=10, ha="center", va="center",
#                 bbox=dict(facecolor="#FFD3B4", edgecolor="black", boxstyle="round,pad=0.5"))
#
#         # Connect this node to its parent with an arrow
#         if parent_x is not None and parent_y is not None:
#             arrow = FancyArrowPatch((parent_x, parent_y), (x, y),
#                                     arrowstyle="-", color="black", lw=1)
#             ax.add_patch(arrow)
#
#         # Adjust y-axis for child nodes
#         next_y = y - 1
#
#         # Recursively plot left and right children with updated x, dx
#         if node.left:
#             plot_node(ax, node.left, x - dx, next_y, dx / 2, x, y)
#         if node.right:
#             plot_node(ax, node.right, x + dx, next_y, dx / 2, x, y)
#
#     # Calculate initial spacing and start the recursive plotting from the root
#     plot_node(ax, root, x=0, y=0, dx=5)
#
#     # Save the plot to file or display it interactively
#     plt.savefig(f"{filename}.png", bbox_inches="tight")
#     print(f"Tree visualization saved as {filename}.png")
#     plt.show()
