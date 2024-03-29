import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


directories = [
    '/Users/anitalarsen/Desktop/P4/500_Sounds'
]

for directory in directories:
        drumFolder = directory

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

# Extract and store filenames
file_paths = features_dataframe['Filename']

# Drop the filename column for normalization
features_for_normalization = features_dataframe.drop(columns=['Filename'])

# Normalize the features
for feature in ['Spectral Centroid', 'Spectral Bandwidth', 'Zero Crossing Rate', 'RMS Energy']:
    min_value = features_for_normalization[feature].min()
    max_value = features_for_normalization[feature].max()
    features_dataframe[feature + ' Normalized'] = (features_for_normalization[feature] - min_value) / (max_value - min_value)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)  # Choose number of components
tsne_components = tsne.fit_transform(features_for_normalization)

# Visualize data after t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c='b', marker='o', edgecolors='k')
plt.title('t-SNE of Sound Files')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Add labels to the points on the scatter plot
#for i, txt in enumerate(file_paths):
#    plt.annotate(txt, (tsne_components[i, 0], tsne_components[i, 1]))

plt.grid(True)
plt.show()