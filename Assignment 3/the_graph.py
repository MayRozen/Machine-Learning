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

def generate_merged_table(data_file, output_file):
    """
    Generate and save a styled table with merged cells and custom pastel colors from CSV data.

    Args:
        data_file (str): Path to the CSV file containing the k-NN results.
        output_file (str): Path to save the output table as an image.
    """
    # Read and decode the data from the CSV file
    try:
        average_results = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: File {data_file} not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Sort the data for consistent reading
    average_results = average_results.sort_values(by=['p', 'k']).reset_index(drop=True)

    # Create the figure and axes for the table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')  # Turn off the axes

    # Add a title to the table
    plt.title("k-NN Classifier Results", fontsize=18, fontname='Comic Sans MS', color='darkblue', pad=20)

    # Prepare the table data
    table_data = []
    row_colors = []
    grouped = average_results.groupby('p', sort=False)

    # Custom pastel colors
    pastel_colors = [
        "#FFCCCC",  # Light pink
        "#CCFFFF",  # Light cyan
        "#FFFFCC",  # Light yellow
        "#CCFFCC",  # Light green
        "#FFCCE5",  # Pale pink
        "#CCCCFF"  # Light lavender
    ]

    # Create table rows for each 'p' value
    for i, (p, group) in enumerate(grouped):
        first_row = True
        for _, row in group.iterrows():
            if first_row:
                table_data.append([p,
                                   row['k'],
                                   f"{row['train_error']:.6f}",
                                   f"{row['test_error']:.6f}",
                                   f"{row['error_difference']:.6f}"])
                first_row = False
            else:
                table_data.append(["",  # Preserve merged rows for 'p'
                                   row['k'],
                                   f"{row['train_error']:.6f}",
                                   f"{row['test_error']:.6f}",
                                   f"{row['error_difference']:.6f}"])
            row_colors.append(pastel_colors[i % len(pastel_colors)])

    # Define column headers
    col_labels = ['p', 'k', 'Train Error', 'Test Error', 'Error Difference']

    # Create the table using Matplotlib
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(list(range(len(col_labels))))

    # Customize table headers and data rows
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Style table headers
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('darkblue')
            cell.set_edgecolor('white')
        else:  # Style data cells
            if col == 0 and cell.get_text().get_text() == "":
                cell.set_facecolor('white')  # White background for merged cells
                cell.set_edgecolor('white')  # Remove border
            else:
                cell.set_facecolor(row_colors[row - 1])
                cell.set_edgecolor('grey')  # Grey borders for visual separation

    # Save the table as an image file
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Styled table saved successfully to {output_file}")
    except Exception as e:
        print(f"Failed to save the table: {e}")

    plt.show()  # Display the table on the screen (optional)
