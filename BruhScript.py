import os
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#import PCAWeight

# Assuming the drumFolder is correctly set to where your .wav files are located
drumFolder = '500_Sounds'

features_list = []

# Iterate over each file in the folder
for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):  # Check if the file is a WAV file
        file_path = os.path.join(drumFolder, filename)

        # Load the audio file
        y, sr = librosa.load(file_path)

        # Extract featuresBruhScript.pyBruhScript.py
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

# Scaling the features before PCA and t-SNE
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_dataframe.iloc[:, 1:])  # Exclude the filename for scaling

# Applying PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_features)

# Identify important axes (features) for PCA by looking at the explained variance ratio
print(f"Explained variance by component: {pca.explained_variance_ratio_}")

# Applying DBSCAN to PCA results
dbscan_pca = DBSCAN(eps=4, min_samples=10)
pca_labels = dbscan_pca.fit_predict(pca_results)

# Count the number of clusters (ignoring noise points labeled as -1)
n_clusters_pca = len(set(pca_labels)) - (1 if -1 in pca_labels else 0)
print(f"Number of clusters identified by DBSCAN in PCA: {n_clusters_pca}")

# Applying t-SNE
tsne = TSNE(n_components=2, perplexity=15)
tsne_results = tsne.fit_transform(scaled_features)

# Applying DBSCAN to t-SNE results
dbscan_tsne = DBSCAN(eps=3, min_samples=10)
tsne_labels = dbscan_tsne.fit_predict(tsne_results)

# Count the number of clusters for t-SNE results
n_clusters_tsne = len(set(tsne_labels)) - (1 if -1 in tsne_labels else 0)
print(f"Number of clusters identified by DBSCAN in t-SNE: {n_clusters_tsne}")

# Plotting
plt.figure(figsize=(14, 7))

# PCA with DBSCAN clusters
plt.subplot(1, 2, 1)
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=pca_labels, cmap='viridis', alpha=0.5)
plt.title('PCA with DBSCAN Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.text(0.05, 0.95, f'Clusters: {n_clusters_pca}', transform=plt.gca().transAxes)

# t-SNE with DBSCAN clusters
plt.subplot(1, 2, 2)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=tsne_labels, cmap='viridis', alpha=0.5)
plt.title('t-SNE with DBSCAN Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster Label')
plt.text(0.05, 0.95, f'Clusters: {n_clusters_tsne}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
pca.fit(scaled_features)

# Accessing the loadings
loadings = pca.components_

print("Loadings for the first principal component:", loadings[0])
print("Loadings for the second principal component:", loadings[1])

