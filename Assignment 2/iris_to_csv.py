import pandas as pd

def iris_to_csv():
    # Define the column names for the Iris dataset
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # Load the Iris dataset from the .txt file
    file_path = r"C:\Users\ASUS\Machine-Learning\Assignment 2\iris.txt"
    iris_data = pd.read_csv(file_path, sep=" ", header=None, names=column_names, engine='python')

    # Drop any extra spaces or columns introduced due to spacing in the text
    iris_data = iris_data.dropna(axis=1, how="all")

    # Display the first few rows of the dataset
    print("iris was successfully converted to a csv file!")
    print(iris_data)

    # Save the dataset to a CSV file if needed
    iris_data.to_csv("iris.csv", index=False)
