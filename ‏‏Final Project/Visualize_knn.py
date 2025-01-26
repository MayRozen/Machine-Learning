import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class KNNVisualization:
    def __init__(self, confusion_matrix, classification_report):
        """
        Initialize KNN Visualization

        Args:
            confusion_matrix (np.array): Confusion matrix from KNN model
            classification_report (str): Classification report
        """
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.pastel_colors = sns.color_palette("pastel")

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
        plt.title('KNN Confusion Matrix', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('knn_confusion_matrix.png')
        plt.close()

    def plot_performance_summary(self):
        """
        Create a bar plot showing precision, recall, and f1-score
        """
        # Parse classification report for visualization
        lines = self.classification_report.split('\n')
        data = [line.split() for line in lines if len(line.split()) > 2]

        categories = [row[0] for row in data[1:]]
        precision = [float(row[1]) for row in data[1:]]
        recall = [float(row[2]) for row in data[1:]]
        f1_score = [float(row[3]) for row in data[1:]]

        plt.figure(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.25

        plt.bar(x - width, precision, width, label='Precision', color=self.pastel_colors[0])
        plt.bar(x, recall, width, label='Recall', color=self.pastel_colors[1])
        plt.bar(x + width, f1_score, width, label='F1-Score', color=self.pastel_colors[2])

        plt.xlabel('Sound Categories', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('KNN Performance Metrics', fontsize=15)
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('knn_performance_metrics.png')
        plt.close()