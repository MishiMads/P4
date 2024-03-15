import os
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming the drumFolder is correctly set to where your .wav files are located
drumFolder = '500_Sounds'

features_list = []

# Iterate over each file in the folder
for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):  # Check if the file is a WAV file
        file_path = os.path.join(drumFolder, filename)

        # Load the audio file
        y, sr = librosa.load(file_path)

        # Extract features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        rms_energy = librosa.feature.rms(y=y).mean()

        # Store the features
        features_list.append({
            'Filename': filename,
            'Spectral Centroid': spectral_centroid,
            'Spectral Bandwidth': spectral_bandwidth,
            'Spectral Rolloff': spectral_rolloff,
            'Zero Crossing Rate': zero_crossing_rate,
            'RMS Energy': rms_energy
        })

# Convert the list of features into a DataFrame
features_dataframe = pd.DataFrame(features_list)

# Scaling the features before PCA
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_dataframe.iloc[:, 1:])  # Exclude the filename for scaling

# Applying PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_features)

# Print explained variance by component
print(f"Explained variance by component: {pca.explained_variance_ratio_}")

# Plotting the PCA component makeup
feature_names = features_dataframe.columns[1:]  # Exclude 'Filename'
pca_components = pca.components_

plt.figure(figsize=(12, 6))
for i, (comp, name) in enumerate(zip(pca_components, ['Component 1', 'Component 2'])):
    plt.subplot(1, 2, i+1)
    plt.barh(range(len(feature_names)), comp, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title(f'Contribution of Features to {name}')
    plt.xlabel('Contribution Weight')
    plt.ylabel('Features')

plt.tight_layout()
plt.show()
