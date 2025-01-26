import os
import numpy as np
import pandas as pd
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AudioPreprocessor:
    def __init__(self, dataset_path):
        """
        Initialize audio preprocessor with dataset path

        Args:
            dataset_path (str): Path to UrbanSound8K dataset
        """
        self.dataset_path = dataset_path
        self.metadata = pd.read_csv(os.path.join(dataset_path, 'metadata', 'UrbanSound8K.csv'))

    def extract_features(self, file_path, max_pad_length=174):
        """
        Extract audio features using MFCC and spectral features

        Args:
            file_path (str): Path to audio file
            max_pad_length (int): Maximum length to pad/truncate features

        Returns:
            np.array: Extracted audio features
        """
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Extract spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]

        # Extract zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

        # Pad or truncate features
        pad_width = max_pad_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max(pad_width, 0))), mode='constant')
        mfccs = mfccs[:, :max_pad_length]

        # Combine features
        features = np.concatenate([
            mfccs,
            np.mean(mfccs, axis=1).reshape(-1, 1),
            np.std(mfccs, axis=1).reshape(-1, 1),
            [np.mean(spectral_centroids)],
            [np.mean(zero_crossing_rate)]
        ]).flatten()

        return features

    def prepare_dataset(self):
        """
        Prepare dataset by extracting features and labels

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        features = []
        labels = []

        # Iterate through audio files
        for index, row in self.metadata.iterrows():
            file_path = os.path.join(
                self.dataset_path,
                f'audio/fold{row["fold"]}',
                row['slice_file_name']
            )

            feature = self.extract_features(file_path)
            features.append(feature)
            labels.append(row['class'])

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test