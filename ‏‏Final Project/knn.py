import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import pairwise_distances
from collections import Counter

class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)

    def preprocess_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        distances = pairwise_distances(X_test, self.X_train)

        for i in range(X_test.shape[0]):
            nearest_neighbors = np.argsort(distances[i])[:self.n_neighbors]
            neighbor_labels = self.y_train[nearest_neighbors]

            label_counts = Counter(neighbor_labels)
            predicted_label = max(label_counts.keys(), key=lambda label: (
                label_counts[label], -np.mean(distances[i, nearest_neighbors][neighbor_labels == label])
            ))
            predictions.append(predicted_label)

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Overall Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(class_report)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\n" + "-" * 50 + "\n")

        self.plot_results(y_test, y_pred)

    def plot_results(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFD700", "#FF69B4"]
        unique_labels = np.unique(y_test)
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

        for i in range(len(y_test)):
            color = color_map[y_test[i]] if y_test[i] == y_pred[i] else "red"
            plt.scatter(i, y_test[i], color=color, label=f"Class {y_test[i]}" if color != "red" else "Misclassified",
                        edgecolors="black")

        plt.title("KNN Classification Results", fontsize=16, fontweight="bold", fontname="Arial")
        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
