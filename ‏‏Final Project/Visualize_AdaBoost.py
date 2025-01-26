import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class AdaBoostVisualization:
    def __init__(self, confusion_matrix, classification_report):
        """
        Initialize AdaBoost Visualization

        Args:
            confusion_matrix (np.array): Confusion matrix from AdaBoost model
            classification_report (str): Classification report
        """
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.pastel_colors = sns.color_palette("pastel")

    def plot_feature_importance(self, feature_importances, feature_names):
        """
        Plot feature importances for AdaBoost

        Args:
            feature_importances (np.array): Importance of each feature
            feature_names (list): Names of features
        """
        plt.figure(figsize=(12, 6))
        feature_indices = np.argsort(feature_importances)

        plt.barh(
            [feature_names[i] for i in feature_indices],
            feature_importances[feature_indices],
            color=[self.pastel_colors[i % len(self.pastel_colors)] for i in range(len(feature_names))]
        )

        plt.title('Feature Importances in AdaBoost', fontsize=15)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('adaboost_feature_importance.png')
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
        plt.title('AdaBoost Confusion Matrix', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('adaboost_confusion_matrix.png')
        plt.close()