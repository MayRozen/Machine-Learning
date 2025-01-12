import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_knn_plot(data_file, output_file):
    """
    Generate and save a k-NN results plot from a CSV file.

    Args:
        data_file (str): Path to the CSV file containing the k-NN results.
        output_file (str): Path to save the output graph as an image.
    """
    # Load data from the given CSV file
    try:
        average_results = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: File {data_file} not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Configure the style and setup the figure
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(12, 8))

    # Plot test errors for different values of k and p
    for p in average_results['p'].unique():
        subset = average_results[average_results['p'] == p]
        plt.plot(subset['k'], subset['test_error'], marker='o', label=f'p={p}')

    # Add title, labels, and legend
    plt.title("k-NN Classifier: Test Errors for k and p", fontsize=18, fontname='Comic Sans MS', color='darkblue')
    plt.xlabel("k (Number of Neighbors)", fontsize=14, fontname='Comic Sans MS')
    plt.ylabel("Test Error", fontsize=14, fontname='Comic Sans MS')
    plt.legend(title="p (Distance Metric)", fontsize=12)

    # Save the plot to the specified file
    try:
        plt.savefig(output_file, dpi=300)
        print(f"Graph saved successfully to {output_file}")
    except Exception as e:
        print(f"Failed to save the graph: {e}")
        return

    # Show the graph (optional, can be removed if not needed)
    plt.show()