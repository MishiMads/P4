import os
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import mplcursors
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd

# Calculates distance between points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Plays closest sound
def play_sound(selected_point):
    scaled_features_array = np.array(scaled_features)
    selected_index = np.argmin([euclidean_distance(selected_point, point) for point in scaled_features_array])
    selected_filename = features_dataframe.iloc[selected_index]['Filename']
    selected_file_path = os.path.join(drumFolder, selected_filename)
    y, sr = librosa.load(selected_file_path, sr=None)
    sd.play(y, sr)
    sd.wait()

drumFolder = '500_Sounds'
features_list = []

for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):
        file_path = os.path.join(drumFolder, filename)
        y, sr = librosa.load(file_path)

        # Extract selected features
        rms_energy = librosa.feature.rms(y=y).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()

        features_list.append({
            'Filename': filename,
            'RMS Energy': rms_energy,
            'Spectral Flatness': spectral_flatness,
        })

# Convert features_list to a DataFrame
features_dataframe = pd.DataFrame(features_list)

# Before scaling, apply log to the selected features to ensure they're all positive
# Exclude the filename column for the logarithm transformation
features_log_transformed = features_dataframe.iloc[:, 1:].apply(np.log)

# Scaling the spread-out log-transformed features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_log_transformed)

# Create window
window = tk.Tk()
window.title("Feature Scatter Plot")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(scaled_features[:, 0], scaled_features[:, 1], alpha=0.5)
ax.set_title('Scatter Plot of RMS Energy and Spectral Flatness')
ax.set_xlabel('Scaled RMS Energy')
ax.set_ylabel('Scaled Spectral Flatness')

mplcursors.cursor(scatter).connect("add", lambda sel: play_sound(np.array([sel.target[0], sel.target[1]])))

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

window.mainloop()
