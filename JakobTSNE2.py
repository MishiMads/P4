import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
import numpy as np

sound_directory = r'C:\Users\jakob\Desktop\100_sounds'
files = []
for file in sound_directory:
    files.append(file)
print(files)


sounds = []

for file_name in os.listdir(sound_directory):
    if file_name.endswith('.wav'):
        file_path = os.path.join(sound_directory, file_name)
        y, sr = librosa.load(file_path)
        sounds.append((y, sr))

#tsne = TSNE(n_components=2)
#reduced_features_tsne = tsne.fit_transform(sounds)
