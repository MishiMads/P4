import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

number_of_features = 3

def extract_features(audio):
    y, sr = librosa.load(audio)
    s_features = []

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    #zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    #rms_energy = librosa.feature.rms(y=y).mean()

    s_features.append(spectral_centroid)
    s_features.append(spectral_bandwidth)
    s_features.append(spectral_rolloff)
    #s_features.append(zero_crossing_rate)
    #s_features.append(rms_energy)

    return s_features


sound_dir = '500_Sounds'

X = []
sound_files = os.listdir(sound_dir)[:500]
for file in sound_files:
    audio_file = os.path.join(sound_dir, file)
    features = extract_features(audio_file)
    X.append(features)

X = np.array(X)
X_log = np.log(X)
X_sqrt = np.sqrt(X)
X_cbrt = np.cbrt(X)

fig, axs = plt.subplots(number_of_features, 1, figsize=(10, 8), sharex='all')

for i in range(number_of_features):
    axs[i].hist(X_log[:, i], bins=20, color='blue', alpha=0.7)
    axs[i].set_title(f'Feature {i + 1}')
    axs[i].set_ylabel('Frequency')

axs[number_of_features - 1].set_xlabel('Feature Values')

pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_log)

plt.figure(figsize=(7, 5))
plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
plt.title('PCA Log() features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.xlim(-1, 1)
plt.ylim(-0.2, 0.2)

plt.tight_layout()
plt.show()
