import os
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
from matplotlib.text import Annotation

# Calculates distance between points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Plays closest sound
def play_sound(selected_index):
    selected_filename = features_dataframe.iloc[selected_index]['Filename']
    selected_file_path = os.path.join(drumFolder, selected_filename)
    y, sr = librosa.load(selected_file_path, sr=None)
    sd.stop()  # Stop any currently playing sounds
    sd.play(y, sr)
    sd.wait()

drumFolder = '500_Sounds'
features_list = []

# Extract selected features
for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):
        file_path = os.path.join(drumFolder, filename)
        y, sr = librosa.load(file_path)

        rms_energy = librosa.feature.rms(y=y).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).mean()
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

        features_list.append({
            'Filename': filename,
            'RMS Energy': rms_energy,
            #'Zero Crossing Rate': zero_crossing_rate,
            'Onset Strength': onset_strength,
            #'Spectral Rolloff': spectral_rolloff,
            'Spectral Centroid': spectral_centroid,
            #'Spectral Contrast': spectral_contrast
        })

# Convert features_list to a DataFrame and preprocess
features_dataframe = pd.DataFrame(features_list)
features_log_transformed = features_dataframe.iloc[:, 1:].apply(np.log)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_log_transformed)

# PCA Transformation
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Create window
window = tk.Tk()
window.title("PCA Scatter Plot of Drum Sounds")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.5, picker=True)

ax.set_title('PCA Scatter Plot of Drum Sounds')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# Tooltip for displaying filenames
tooltip = Annotation("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="w"),
                     arrowprops=dict(arrowstyle="->"))
tooltip.set_visible(False)
ax.add_artist(tooltip)

def on_pick(event):
    if event.artist == scatter:
        ind = event.ind[0]  # Get the index of the clicked point
        play_sound(ind)

def update_tooltip(event):
    vis = tooltip.get_visible()
    if event.inaxes == ax:
        contains, ind = scatter.contains(event)
        if contains:
            index = ind['ind'][0]
            x, y = pca_features[index]
            tooltip.xy = (x, y)
            tooltip.set_text(features_dataframe['Filename'].iloc[index])
            tooltip.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                tooltip.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect("motion_notify_event", update_tooltip)

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

window.mainloop()
