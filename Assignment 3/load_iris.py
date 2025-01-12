import pandas as pd
import numpy as np


def load_iris():
    """
    Load the Iris dataset from a CSV file and prepare it for training.
    """
    # File path to the iris dataset
    file_path = r"C:\Users\ASUS\Machine-Learning\Assignment 3\iris.csv"

    # Load the dataset with column names
    column_names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
    data_file = pd.read_csv(file_path, names=column_names, header=0)  # Assumes the CSV file has headers

    # Check for missing data or an empty dataset
    if data_file.empty:
        raise ValueError("The CSV file is empty or not formatted correctly.")
    if data_file.shape[1] < 5:
        raise ValueError("The dataset does not have the required columns.")

    # Select "PetalLength" and "PetalWidth" columns
    x = data_file[["PetalLength", "PetalWidth"]].values

    # Validate data: ensure all values are numeric
    x_valid = pd.DataFrame(x, columns=["PetalLength", "PetalWidth"]).apply(pd.to_numeric,
                                                                           errors="coerce").dropna().values
    if x_valid.shape[0] == 0:
        raise ValueError("No valid numeric data found in the dataset.")

    # Encode the target labels
    y = data_file["Species"].values
    label_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    y_encoded = np.array([label_mapping[label] for label in y if label in label_mapping])

    if y_encoded.shape[0] == 0:
        raise ValueError("No valid labels found in the dataset.")

    print(f"Dataset loaded successfully: {x_valid.shape[0]} samples, {x_valid.shape[1]} features.")
    return x_valid, y_encoded