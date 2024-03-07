import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

sound_files_dir = "C:/Users/rglus/OneDrive/Skrivebord/Kurser/4. semester/P4/Tilsendte materialer/A_kicks/At variablere"

sound_files = [os.path.join(sound_files_dir, file) for file in os.listdir(sound_files_dir) if file.endswith(".wav")]

#feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0, 0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0, 0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0, 0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0, 0]
    rmse = librosa.feature.rms(y=y)[0, 0]

    return [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, rmse]

#characteristics dictionary
char_dict = {
    '028600-STKEV1 Kick - 189': ['tight', 'punchy', 'warm'],
    '028601-STKEV1 Kick - 190 - sub1': ['bright', 'sharp', 'metallic'],
    '028601-STKEV1 Kick - 190 - sub2': ['tight', 'punchy', 'warm'],
    '028602-STKEV1 Kick - 191': ['muddy', 'punchy', 'warm'],
    '028603-STKEV1 Kick - 192': ['muddy', 'punchy', 'warm'],
    '028604-STKEV1 Kick - 193': ['tight', 'crisp', 'wood']
}

#PCA of features
feat_matrix = np.array([extract_features(file_path) for file_path in sound_files])

pca_feat = PCA(n_components=2)
features_2d = pca_feat.fit_transform(feat_matrix)

#PCA of characteristics
char_matrix = np.array([char_dict.get(os.path.basename(file).split('.')[0], []) for file in sound_files], dtype=object)

print(char_matrix)

pca_char = PCA(n_components=2)

feat_2d = pca_char.fit_transform(char_matrix)

#mean of sound
mean_coordinates = (features_2d + characteristics_2d) / 2

#plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3D')
ax.scatter(mean_coordinates[:, 0], mean_coordinates[:, 1], characteristics_2d[:, 1], c='r', marker='o')

ax.set_xlabel('PCA Feature 1')
ax.set_ylabel('PCA Feature 2')
ax.set_xlabel('PCA Characteristics')

plt.show()