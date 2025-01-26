import os
from preprocessing.data_loader import AudioPreprocessor
from models.knn_classifier import KNNClassifier
from models.svm_classifier import SVMClassifier
from models.clustering import AudioClustering
from models.linear_regression import AudioLinearRegression
from models.adaboost_classifier import AdaBoostAudioClassifier


def main():
    # Path to UrbanSound8K dataset
    dataset_path = 'path/to/UrbanSound8K'

    # Initialize preprocessor
    preprocessor = AudioPreprocessor(dataset_path)

    # Prepare dataset
    X_train, X_test, y_train, y_test = preprocessor.prepare_dataset()

    # Initialize and evaluate models
    models = {
        'KNN': KNNClassifier(),
        'SVM': SVMClassifier(),
        'AdaBoost': AdaBoostAudioClassifier()
    }

    results = {}

    # Train and evaluate classification models
    for name, model in models.items():
        model.train(X_train, y_train)
        results[name] = model.evaluate(X_test, y_test)
        print(f"{name} Results:\n{results[name]['classification_report']}\n")

    # Clustering
    clustering = AudioClustering()
    clustering_score = clustering.evaluate_clustering(X_train)
    print(f"Clustering Silhouette Score: {clustering_score}")

    # Optional: Linear Regression (example with spectral features)
    lr_model = AudioLinearRegression()
    lr_model.train(X_train[:, :10], y_train)  # Using first 10 features as example
    lr_results = lr_model.evaluate(X_test[:, :10], y_test)
    print(f"Linear Regression Results:\n{lr_results}")


if __name__ == '__main__':
    main()