import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import sounddevice as sd


sound_dir = '500_Sounds'
feature_list = []


def extract_feature1(audio):
    y, sr = librosa.load(audio)

    return librosa.feature.spectral_rolloff(y=y, sr=sr).mean()


def extract_feature2(audio):
    y, sr = librosa.load(audio)

    return librosa.feature.rms(y=y).mean()


X = []
Y = []
sound_files = os.listdir(sound_dir)[:500]
for file in sound_files:
    audio_file = os.path.join(sound_dir, file)
    feature1 = extract_feature1(audio_file)
    X.append(feature1)
    feature2 = extract_feature2(audio_file)
    Y.append(feature2)
    feature_list.append([feature1, feature2])

X_log = np.log(X)
Y_log = np.log(Y)

# plt.figure(figsize=(7, 5))
# plt.scatter(X, Y, alpha=0.5)
# plt.title('2 Features')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_list)

plt.figure(figsize=(7, 5))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
plt.title('Plot')
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')

plt.show()
