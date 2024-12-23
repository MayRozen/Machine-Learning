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

    x = data_file.iloc[:, [1, 2]].values
    print(x)

    x_vector = x.flatten()
    print(x_vector)

    y = data_file.iloc[:, -1].values  # Assuming the last column contains the labels
    y_binary = np.where(y == 'Iris-setosa', -1, 1)

    return x_vector, y_binary