import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def load_iris():
    # File path for your Iris dataset
    file_path = r"C:\Users\ASUS\Machine-Learning\Assignment 2\iris.csv"

    # Load the dataset
    column_names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
    data_file = pd.read_csv(file_path, names=column_names)

    # Inspect the first few rows of the dataset
    print("iris.csv dataset was successfully loaded!")

    if data_file.shape[1] < 3:
        raise ValueError("Data does not have enough columns for selection.")
    if data_file.empty:
        raise ValueError("The DataFrame 'data' is empty.")

    # Selecting specific columns and cleaning data
    x = data_file.iloc[1:, [1, 2]].values

    x_valid = []
    for row in x:
        try:
            x_valid.append([float(value) for value in row])
        except ValueError:
            print(f"Skipping invalid row: {row!r}")  # Log if invalid values are encountered

    if len(x_valid) == 0:
        raise ValueError("No valid rows found in dataset after cleaning.")

    x_vector = np.array(x_valid)

    # Preparing labels for all three species (Setosa, Versicolor, Virginica)
    y = data_file.iloc[1:, -1].values  # Extract labels (species column)

    # Map the species to numerical values
    y_encoded = np.zeros_like(y, dtype=int)
    y_encoded[y == "Iris-setosa"] = 0  # Setosa -> 0
    y_encoded[y == "Iris-versicolor"] = 1  # Versicolor -> 1
    y_encoded[y == "Iris-virginica"] = 2  # Virginica -> 2
    print(y_encoded)

    return x_vector, y_encoded