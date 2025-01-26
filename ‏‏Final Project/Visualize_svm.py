import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SVMVisualization:
    def __init__(self, confusion_matrix, classification_report):
        """
        Initialize SVM Visualization

        Args:
            confusion_matrix (np.array): Confusion matrix from SVM model
            classification_report (str): Classification report
        """
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.pastel_colors = sns.color_palette("pastel")

    def plot_support_vectors(self, X, y):
        """
        Visualize Support Vectors (2D Projection)

        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
        """
        plt.figure(figsize=(12, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
        plt.title('SVM Support Vectors Projection', fontsize=15)
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.colorbar(label='Sound Category')
        plt.tight_layout()
        plt.savefig('svm_support_vectors.png')
        plt.close()

    def plot_confusion_matrix(self):
        """
        Create a pastel-colored confusion matrix heatmap
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            cmap=self.pastel_colors,
            cbar_kws={'label': 'Prediction Count'}
        )
        plt.title('SVM Confusion Matrix', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('svm_confusion_matrix.png')
        plt.close()