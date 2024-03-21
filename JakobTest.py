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


sound_dir = '500_Sounds'


def extract_feature1(audio):
    y, sr = librosa.load(audio)

    return librosa.feature.zero_crossing_rate(y).mean()


def extract_feature2(audio):
    y, sr = librosa.load(audio)

    return librosa.feature.spectral_flatness(y=y).mean()


X = []
Y = []
sound_files = os.listdir(sound_dir)[:500]
for file in sound_files:
    audio_file = os.path.join(sound_dir, file)
    feature1 = extract_feature1(audio_file)
    X.append(feature1)
    feature2 = extract_feature2(audio_file)
    Y.append(feature2)

plt.figure(figsize=(7, 5))
plt.scatter(X, Y, alpha=0.5)
plt.title('Zero Crossing Rate vs Spectral Flatness')
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Spectral Flatness')
plt.show()
