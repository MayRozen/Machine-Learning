from iris_to_csv import iris_to_csv
from load_iris import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from the_graph import generate_knn_plot, generate_merged_table


def knn_classifier(training_data, training_labels, test_data, k, p):
    """
    In-house k-Nearest Neighbors implementation.
    """
    distances = cdist(test_data, training_data, metric='minkowski', p=p)  # Compute distances using Minkowski metric
    neighbors_indices = np.argsort(distances, axis=1)[:, :k]  # Sort and get indices of k nearest neighbors
    neighbor_labels = training_labels[neighbors_indices]  # Get labels of k nearest neighbors
    predictions = np.array([np.bincount(labels).argmax() for labels in neighbor_labels])  # Predict the majority class
    return predictions


# Step 1: Convert Iris dataset from TXT to CSV
iris_to_csv()  # This will create the `iris.csv` file

# Step 2: Load the dataset from the newly created CSV file
data, labels = load_iris()  # Loads and prepares `iris.csv`

ks = [1, 3, 5, 7, 9]
ps = [1, 2, np.inf]
results = []

# Run the algorithm 100 times to calculate empirical results
for iteration in range(100):  # Repeat the process 100 times
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5, stratify=labels)

    for k in ks:
        for p in ps:
            # Predict on the training set
            train_predictions = knn_classifier(train_data, train_labels, train_data, k, p)
            train_error = 1 - accuracy_score(train_labels, train_predictions)

            # Predict on the test set
            test_predictions = knn_classifier(train_data, train_labels, test_data, k, p)
            test_error = 1 - accuracy_score(test_labels, test_predictions)

            # Store the results: k, p, errors, and their difference
            results.append((k, p, train_error, test_error, test_error - train_error))

# Compute the average empirical results
df_results = pd.DataFrame(results, columns=['k', 'p', 'train_error', 'test_error', 'error_difference'])
average_results = df_results.groupby(['k', 'p']).mean().reset_index()

# Save the average results to a CSV file
data_file = "average_results.csv"
average_results.to_csv(data_file, index=False)


# Print the formatted average empirical results
def print_formatted_average_results(average_results):
    """
    Print the formatted average results by iterating over each value of 'p'.
    """
    print("Empirical Average Results:")  # Title for the output

    # Sort the data by 'p' and 'k'
    average_results = average_results.sort_values(by=['p', 'k'])

    for p in average_results['p'].unique():
        print(f"p: {p}")
        subset = average_results[average_results['p'] == p]
        for _, row in subset.iterrows():
            print(
                f"k: {row['k']:<4} Train Error: {row['train_error']:<10.6f} Test Error: {row['test_error']:<10.6f} Error Difference: {row['error_difference']:<10.6f}")
        print()  # Add a blank line between each group of p


# Display the formatted average results in the required format
print_formatted_average_results(average_results)

if __name__ == "__main__":
    # Path to the CSV file with results
    data_file = "average_results.csv"

    # Path to save the output graph
    output_image1_file = "results_graph.png" # First image (graph)
    output_image2_file = "results_table.png" # Second image (table)

    # Call the function to generate the graph
    generate_knn_plot(data_file, output_image1_file)
    generate_merged_table(data_file, output_image2_file)

    print(f"The graph has been saved to {output_image1_file}")
    print(f"The table has been saved to {output_image2_file}")