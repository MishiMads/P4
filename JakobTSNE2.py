import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
import numpy as np

sound_directory = r'C:\Users\jakob\Desktop\A_kicks\subdir1'


def plot_sound_characteristics(directory):
    plt.figure(figsize=(8, 6))
    s_features = None

    for file_name in directory:
        if file_name.endswith('.wav'):
            sound_path = os.path.join(directory, file_name)

            y, sr = librosa.load(sound_path)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
            rms_energy = librosa.feature.rms(y=y).mean()

            s_features = np.vstack((s_features, [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, rms_energy]))

            if s_features is None:
                s_features = features.reshape(1, -1)
            else:
                s_features = np.vstack((s_features, features))

    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    features_embedded = tsne.fit_transform(s_features)

    plt.figure(figsize=(8, 6))
    for i, feature in enumerate(features_embedded):
        plt.scatter(feature[0], feature[1], label=os.listdir(directory)[i]) # Using file names as labels    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Sound Characteristics')
    plt.grid(True)
    plt.show()


plot_sound_characteristics(sound_directory)
