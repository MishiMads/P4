import os
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # Importing seaborn for the heatmap
import numpy as np

drumFolder = '500_Sounds'
features_list = []

for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):
        file_path = os.path.join(drumFolder, filename)
        y, sr = librosa.load(file_path)

        # Extract features and summarize as needed
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        rms_energy = librosa.feature.rms(y=y).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).mean()  # Mean across frequencies then across frames
        onset_env = librosa.onset.onset_strength(y=y, sr=sr).mean()

        features_list.append({
            'Filename': filename,
            'Spectral Centroid': spectral_centroid,
            'Spectral Bandwidth': spectral_bandwidth,
            'Spectral Rolloff': spectral_rolloff,
            'Zero Crossing Rate': zero_crossing_rate,
            'RMS Energy': rms_energy,
            'Spectral Flatness': spectral_flatness,
            'Spectral Contrast': spectral_contrast,
            'Onset Envelope': onset_env
        })

# Convert features_list to a DataFrame
features_dataframe = pd.DataFrame(features_list)

# Remove the 'Filename' column for correlation analysis
features = features_dataframe.drop(['Filename'], axis=1)

# Calculate the correlation matrix
correlation_matrix = features.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Audio Features')
plt.show()

sns.pairplot(features)

plt.show()
