import os
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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

# Selecting features for t-SNE and PCA
features = features_dataframe[['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff', 'Zero Crossing Rate', 'RMS Energy']]

# Applying PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(features)

# Applying DBSCAN
# These parameters (eps and min_samples) might need adjustment based on your data
dbscan = DBSCAN(eps=4, min_samples=3).fit(pca_results)

# Getting cluster labels (note: -1 means outlier)
cluster_labels = dbscan.labels_

# Visualizing PCA with DBSCAN clusters
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title('PCA with DBSCAN Clusters')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# If you wish to compare with t-SNE
# Applying t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(features)

# Re-applying DBSCAN on t-SNE results for comparison
dbscan_tsne = DBSCAN(eps=3, min_samples=2).fit(tsne_results)
tsne_labels = dbscan_tsne.labels_

plt.subplot(1, 2, 2)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=tsne_labels, cmap='viridis', alpha=0.5)
plt.title('t-SNE with DBSCAN Clusters')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.show()