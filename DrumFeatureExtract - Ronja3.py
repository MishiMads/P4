import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas

directories = [
    '/Users/mads/Desktop/MED5/A_kicks/subdir1',
    'Jakob',
    'Anita',
    'C:/Users/rglus/OneDrive/Skrivebord/Kurser/4. semester/P4/Tilsendte materialer/A_kicks/At variablere',
    'Anton',
]

for directory in directories:
    if os.path.exists(directory):
        drumFolder = directory
    else:
        print('No Bueno')


features_list = []
characteristics_list = []

characteristics_mapping = {
'028600-STKEV1 Kick - 189.wav': {'Tight': 1, 'Punchy': 1, 'Warm': 1},
'028600-STKEV2 Kick - 190.wav': {'Bright': 1, 'Sharp': 1, 'Metallic': 1, 'Tight': 0.5, 'Punchy': 0.5, 'Warm': 0.5},
'028600-STKEV3 Kick - 191.wav': {'Muddy': 1, 'Punchy': 1, 'Warm': 1},
'028600-STKEV4 Kick - 192.wav': {'Muddy': 1, 'Punchy': 1, 'Warm': 1},
'028600-STKEV5 Kick - 193.wav': {'Tight': 1, 'Punchy': 1, 'Crisp': 1, 'Wood': 1}
}

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

        if drumFolder == 'C:/Users/rglus/OneDrive/Skrivebord/Kurser/4. semester/P4/Tilsendte materialer/A_kicks/At variablere' and filename in characteristics_mapping:
            file_characteristics = characteristics_mapping[filename]
        else:
            file_characteristics = {'Tight': 0.0, 'Punchy': 0.0, 'Warm': 0.0, 'Bright': 0.0, 'Sharp': 0.0, 'Metallic': 0.0, 'Muddy': 0.0, 'Crisp': 0.0, 'Wood': 0.0}

        #tightness = 0.2
        #punch = 0.2
        #warmth = 0.2
        #brightness = 0.2
        #sharpness = 0.2
        #metallic = 0.2
        #muddyness = 0.2
        #crispyness = 0.2
        #woodyness = 0.2
        #boominess = 0.2
        #resonance = 0.2

        #Characteristics
        characteristics_list.append({
            'Filename': filename,
            **file_characteristics,
            #'Tight': tightness,
            #'Punchy': punch,
            #'Warm': warmth,
            #'Bright': brightness,
            #'Sharp': sharpness,
            #'Metallic': metallic,
            #'Muddy': muddyness,
            #'Crisp': crispyness,
            #'Wood': woodyness,
            #'Boomy': boominess,
            #'Resonant': resonance
        })

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
characteristics_dataframe = pandas.DataFrame(characteristics_list)

# Normalize the features
for feature in ['Spectral Centroid', 'Spectral Bandwidth', 'Zero Crossing Rate', 'RMS Energy']:
    min_value = features_dataframe[feature].min()
    max_value = features_dataframe[feature].max()
    features_dataframe[feature + ' Normalized'] = (features_dataframe[feature] - min_value) / (max_value - min_value)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    features_dataframe['Spectral Centroid Normalized'],
    features_dataframe['Spectral Bandwidth Normalized'],
    c=characteristics_dataframe['Tight'],
    #characteristics_dataframe['Tight'],
    #characteristics_dataframe['Punchy'],
    #characteristics_dataframe['Warm'],
    #characteristics_dataframe['Bright'],
    #characteristics_dataframe['Sharp'],
    #characteristics_dataframe['Metallic'],
    #characteristics_dataframe['Muddy'],
    #characteristics_dataframe['Crisp'],
    #characteristics_dataframe['Wood'],
    cmap='viridis',
    s=characteristics_dataframe['Punchy']*100,
    alpha=0.7
)

ax.set_xlabel('Normalized Spectral Centroid')
ax.set_ylabel('Normalized Spectral Bandwidth')
ax.set_zlabel('Characteristics')

# Now, plotting two features against each other in a scatter plot
#plt.figure(figsize=(10, 8))
#plt.scatter(
#    features_dataframe['Spectral Centroid Normalized'],
#    features_dataframe['Spectral Bandwidth Normalized'],
#    characteristics_dataframe['Tight'],
#    characteristics_dataframe['Punchy'],
#    characteristics_dataframe['Warm'],
#    characteristics_dataframe['Bright'],
#    characteristics_dataframe['Sharp'],
#    characteristics_dataframe['Metallic'],
#    characteristics_dataframe['Muddy'],
#    characteristics_dataframe['Crisp'],
#    characteristics_dataframe['Wood'],
#    cmap='viridis',
#    alpha=0.5
#)

#plt.title('Normalized Spectral Bandwidth vs Spectral Bandwidth')
#plt.xlabel('Normalized Spectral Centroid')
#plt.ylabel('Normalized Spectral Bandwidth')
#plt.colorbar(label='Tight')
#plt.grid(True)

cbar = plt.colorbar(sc, ax=ax, label='Tight')
plt.title('3D Scatter Plot - Spectral Features vs Characteristics')

plt.show()

#for feature in features_list:
#    print(feature)

#plt.show()