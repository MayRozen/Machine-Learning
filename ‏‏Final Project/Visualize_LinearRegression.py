import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class LinearRegressionVisualization:
    def __init__(self, y_true, y_pred):
        """
        Initialize Linear Regression Visualization

        Args:
            y_true (np.array): True target values
            y_pred (np.array): Predicted target values
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.pastel_colors = sns.color_palette("pastel")

    def plot_regression_line(self):
        """
        Plot regression line with actual vs predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.y_true,
            self.y_pred,
            color=self.pastel_colors[0],
            alpha=0.7
        )

        # Perfect prediction line
        plt.plot(
            [self.y_true.min(), self.y_true.max()],
            [self.y_true.min(), self.y_true.max()],
            'r--',
            lw=2
        )

        plt.title('Actual vs Predicted Values', fontsize=15)
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.tight_layout()
        plt.savefig('linear_regression_plot.png')
        plt.close()

    def plot_residuals(self):
        """
        Plot residuals to understand model performance
        """
        residuals = self.y_true - self.y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.y_pred,
            residuals,
            color=self.pastel_colors[1],
            alpha=0.7
        )
        plt.axhline(y=0, color='r', linestyle='--')

        plt.title('Residual Plot', fontsize=15)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.tight_layout()
        plt.savefig('linear_regression_residuals.png')
        plt.close()