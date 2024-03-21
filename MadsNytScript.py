import os
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

drumFolder = '500_Sounds'
features_list = []

for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):
        file_path = os.path.join(drumFolder, filename)
        y, sr = librosa.load(file_path)

        # Extract features and summarize multidimensional features as needed
        features = {
            'Filename': filename,
            'Spectral Centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
            'Spectral Bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
            'Spectral Rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
            'Zero Crossing Rate': librosa.feature.zero_crossing_rate(y).mean(),
            'RMS Energy': librosa.feature.rms(y=y).mean(),
            'Spectral Flatness': librosa.feature.spectral_flatness(y=y).mean(),
            'Spectral Contrast': librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).mean(),
            # Mean across frequencies then across frames
            'Onset Envelope': librosa.onset.onset_strength(y=y, sr=sr).mean(),
        }
        features_list.append(features)

features_dataframe = pd.DataFrame(features_list)

# Preparing for logarithmic transformation (excluding 'Filename' column)
features_for_transformation = features_dataframe.drop('Filename', axis=1)

# Ensure all values are positive by adding a small constant
features_for_transformation += 1e-6

# Applying logarithmic transformation
log_transformed_features = np.log(features_for_transformation)

# Scaling the log-transformed features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(log_transformed_features)

# Applying PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(log_transformed_features)

# Plotting PCA results
plt.figure(figsize=(7, 5))
plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
plt.title('PCA Results on Log-transformed and Scaled Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Print explained variance
print(f"Explained variance by component: {pca.explained_variance_ratio_}")

# Print loadings
print("Loadings for the first principal component:", pca.components_[0])
print("Loadings for the second principal component:", pca.components_[1])
