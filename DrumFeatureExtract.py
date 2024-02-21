import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas


drumFolder = '/Users/mads/Desktop/MED5/A_kicks/subdir1'

#drumSample = '/Users/mads/Desktop/MED5/A_kicks/subdir1/000000-KICK_ARBLICK.wav'
#y, sr = librosa.load(drumSample)

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
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        rms_energy = librosa.feature.rms(y=y).mean()

        # Store the features
        features_list.append({
            'Filename': filename,
            'Spectral Centroid': spectral_centroid,
            'Spectral Bandwidth': spectral_bandwidth,
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

# Now, plotting two features against each other in a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(features_dataframe['Spectral Centroid Normalized'], features_dataframe['Spectral Bandwidth Normalized'], alpha=0.5)
plt.title('Normalized Spectral Centroid vs Spectral Bandwidth')
plt.xlabel('Normalized Spectral Centroid')
plt.ylabel('Normalized Spectral Bandwidth')
plt.grid(True)
plt.show()

bæh = 0