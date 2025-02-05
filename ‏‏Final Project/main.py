import os
import zipfile
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn import  KNNClassifier
from svm import  SVMClassifier
from Clustering import ClusteringClassifier


# Audio Preprocessor Class
class AudioPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def extract_features(self, audio_path):
        """
        Extracting MFCC features from an audio file
        """
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)

    def prepare_dataset(self):
        """
        Preparing dataset by extracting features from audio files
        """
        X = []
        y = []
        # Extract features from each audio file
        for label in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith('.wav'):
                        audio_path = os.path.join(label_path, file)
                        features = self.extract_features(audio_path)
                        X.append(features)
                        y.append(label)
        X = np.array(X)
        y = np.array(y)

        # Encoding labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to unzip the dataset
def unzip_dataset(zip_path, extract_to='UrbanSound8K'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset extracted to {extract_to}")


def main():
    # Path to UrbanSound8K dataset ZIP
    zip_path = 'archive.zip'

    # Unzip the dataset
    unzip_dataset(zip_path)

    # Path to extracted UrbanSound8K dataset
    dataset_path = 'UrbanSound8K'

    # Initialize preprocessor
    preprocessor = AudioPreprocessor(dataset_path)

    # Prepare dataset
    X_train, X_test, y_train, y_test = preprocessor.prepare_dataset()

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    models = {
        # 'KNN': KNNClassifier(),  # KNN model
        # 'SVM': SVMClassifier(),  # SVM model
        'Clustering': ClusteringClassifier(),  # Clustering model
    }

    results = {}

    # Train and evaluate classification models
    for name, model in models.items():
        print(f"Training {name} model...")

        if name == 'Clustering':
            # Clustering model only needs X_train
            model.train(X_train)
            model.evaluate_clustering(X_train, X_test)  # Pass both X_train and X_test
        else:
            # Other models (like KNN, SVM) need both X_train and y_train
            model.train(X_train, y_train)
            model.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()
