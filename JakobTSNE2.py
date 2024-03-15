import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
import numpy as np


def extract_features(audio):
    y, sr = librosa.load(audio)
    s_features = []

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    spectral_flux

    s_features.append(spectral_centroid)
    s_features.append(spectral_bandwidth)
    s_features.append(spectral_rolloff)
    s_features.append(zero_crossing_rate)

    return s_features


sound_dir = '500_Sounds'

X = []
sound_files = os.listdir(sound_dir)[:500]
for file in sound_files:
    audio_file = os.path.join(sound_dir, file)
    features = extract_features(audio_file)
    X.append(features)

X = np.array(X)
print(X.shape)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='.', color='b')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
