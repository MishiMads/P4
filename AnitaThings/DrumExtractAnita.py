import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


directories = [
    '/Users/mads/Desktop/MED5/A_kicks/subdir1',
    '/Users/jakob/Desktop/A_kicks/subdir1',
    '/Users/anitalarsen/Downloads/A_kicks/subdir1',
    'C:\Mediologi\P4\Rapport\A_kicks\subdir1',
    'C:/Users/rglus/OneDrive/Skrivebord/Kurser/4. semester/P4/Tilsendte materialer/A_kicks/subdir287',
]

for directory in directories:
    if os.path.exists(directory):
        drumFolder = directory
    else:
        print('No Bueno')


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
features_dataframe = pandas.DataFrame(features_list)

# Normalize the features
for feature in ['Spectral Centroid', 'Spectral Bandwidth', 'Zero Crossing Rate', 'RMS Energy']:
    min_value = features_dataframe[feature].min()
    max_value = features_dataframe[feature].max()
    features_dataframe[feature + ' Normalized'] = (features_dataframe[feature] - min_value) / (max_value - min_value)

# Apply PCA
pca = PCA(n_components=2)  # Choose number of components
principal_components = pca.fit_transform(features_dataframe.iloc[:, 4:])  # Use normalized features for PCA

# Visualize data after PCA
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c='b', marker='o', edgecolors='k')
plt.title('PCA of Sound Files')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()