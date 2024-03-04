import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
import numpy as np

directory = r'C:/Users/rglus/OneDrive/Skrivebord/Kurser/4. semester/P4/Tilsendte materialer/A_kicks/At variablere',

class Soundfile:
    def __init__(self, path, filename, tight, punchy, warm, bright, sharp, metallic, muddy, crisp, wood):
        self.path = path
        self.filename = filename
        self.tight = tight
        self.punchy = punchy
        self.warm = warm
        self.bright = bright
        self.sharp = sharp
        self.metallic = metallic
        self.muddy = muddy
        self.crisp = crisp
        self.wood = wood

Sound1 = Soundfile('RonjaLyde/028600-STKEV1 Kick - 189.wav', "sound1", True, True, True, False, False, False, False, False, False)
Sound2 = Soundfile('RonjaLyde/028601-STKEV1 Kick - 190.wav', "sound2", True, True, True, True, True, True, False, False, False)
Sound3 = Soundfile('RonjaLyde/028602-STKEV1 Kick - 191.wav', "sound3", False, True, True, False, False, False, True, False, False)
Sound4 = Soundfile('RonjaLyde/028603-STKEV1 Kick - 192.wav', "sound4", False, True, True, False, False, False, True, False, False)
Sound5 = Soundfile('RonjaLyde/028604-STKEV1 Kick - 193.wav', "sound5", True, True, False, False, False, False, False, True, True)

lyde = [Sound1, Sound2, Sound3, Sound4, Sound5]

for filename in os.listdir(lyde):
    if filename.endswith('.wav'):
        file_path = os.path.join(lyde, filename)

        y, sr = librosa.load(file_path)

        spectral_centroid = librosa.feature.spectral_centrod(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        rms_energy = librosa.feature.rms(y=y).mean()

        features_list.append({
            'Filename': filename,
            'Spectral Centroid': spectral_centroid,
            'Spectral Bandwidth': spectral_bandwidth,
            'Spectral Rolloff': spectral_rolloff,
            'Zero Crossing Rate': zero_crossing_rate,
            'RMS Energy': rms_energy
        })

features_dataframe = pandas.DataFrame(features_list)

for feature in ['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff', 'Zero Crossing Rate', 'RMS Energy']:
    min_value = features_dataframe[feature].min()
    max_value = features_dataframe[feature].max()
    features_dataframe[feature + 'Normalized'] = (features_dataframe[feature] - min_value) / (max_value - min_value)

for directory in directories:
    if os.path.exists(directory):
        drumFolder = directory
    else:
        print('No Bueno')

features_list = []
characteristics_list = []

characteristics_mapping = {
'028600-STKEV1 Kick - 189.wav': {'Tight': 0.15, 'Punchy': 0.15, 'Warm': 0.15},
'028600-STKEV2 Kick - 190.wav': {'Bright': 0.15, 'Sharp': 0.15, 'Metallic': 0.15, 'Tight': 0.15, 'Punchy': 0.15, 'Warm': 0.15},
'028600-STKEV3 Kick - 191.wav': {'Muddy': 0.15, 'Punchy': 0.15, 'Warm': 0.15},
'028600-STKEV4 Kick - 192.wav': {'Muddy': 0.15, 'Punchy': 0.15, 'Warm': 0.15},
'028600-STKEV5 Kick - 193.wav': {'Tight': 0.15, 'Punchy': 0.15, 'Crisp': 0.15, 'Wood': 0.15}
}

for filename, characteristics in characteristics_mapping.items():
    characteristics_mapping[filename] = {
        'Tight': characteristics.get('Tight', 0),
        'Punchy': characteristics.get('Punchy', 0),
        'Warm': characteristics.get('Warm', 0),
        'Bright': characteristics.get('Bright', 0),
        'Sharp': characteristics.get('Sharp', 0),
        'Metallic': characteristics.get('Metallic', 0),
        'Muddy': characteristics.get('Muddy', 0),
        'Crisp': characteristics.get('Crisp', 0),
        'Wood': characteristics.get('Wood', 0)
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

        #Characteristics
        characteristics_list.append({
            'Filename': filename,
            **file_characteristics,
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