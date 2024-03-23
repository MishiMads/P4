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
from matplotlib.text import Annotation


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
    sd.stop()  # Stop any currently playing sounds
    sd.play(y, sr)
    sd.wait()


drumFolder = r'/Users/anitalarsen/Desktop/P4/500_Sounds'
features_list = []

for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):
        file_path = os.path.join(drumFolder, filename)
        y, sr = librosa.load(file_path)

        # Extract selected features
        rms_energy = librosa.feature.rms(y=y).mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
        rms_energy = librosa.feature.rms(y=y).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).mean()
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()

        features_list.append({
            'Filename': filename,
            'RMS Energy': rms_energy,
            'Zero Crossing Rate': zero_crossing_rate,
            #'Onset Strength': onset_strength,
            #'Spectral Rolloff': spectral_rolloff,
            #'Spectral Contrast': spectral_contrast,
        })

# Convert features_list to a DataFrame
features_dataframe = pd.DataFrame(features_list)

# Before scaling, apply log to the selected features to ensure they're all positive
# Exclude the filename column for the logarithm transformation
features_log_transformed = features_dataframe.iloc[:, 1:].apply(np.log)

# Scaling the spread-out log-transformed features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_log_transformed)

# Convert features_list to a DataFrame
features_dataframe = pd.DataFrame(features_list)

# Assume the first two feature names after 'Filename' are what we want for the x and y axes
feature_x_name = features_dataframe.columns[1]  # This will be 'RMS Energy' based on your extraction
feature_y_name = features_dataframe.columns[2]  # This will be 'Zero Crossing Rate'

# Create window
window = tk.Tk()
window.title("Feature Scatter Plot")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(scaled_features[:, 0], scaled_features[:, 1], alpha=0.5, picker=5)  # Enable picker with a tolerance
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

# Set titles and labels dynamically based on the selected features
ax.set_title(f'Scatter Plot of Scaled {feature_x_name} vs Scaled {feature_y_name}')
ax.set_xlabel(f'Scaled {feature_x_name}')
ax.set_ylabel(f'Scaled {feature_y_name}')

# Initialize a text annotation for displaying filenames, but set it to be invisible for now
tooltip = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"))
tooltip.set_visible(False)


def on_pick(event):
    if event.artist == scatter:
        ind = event.ind[0]  # Get the index of the clicked point
        selected_point = np.array([scaled_features[ind, 0], scaled_features[ind, 1]])
        play_sound(selected_point)


def update_tooltip(event):
    vis = tooltip.get_visible()
    if event.inaxes == ax:
        contains, ind = scatter.contains(event)
        if contains:
            index = ind['ind'][0]
            x, y = scatter.get_offsets()[index]
            tooltip.xy = (x, y)
            tooltip.set_text(features_dataframe['Filename'].iloc[index])
            tooltip.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                tooltip.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect('pick_event', on_pick)

# Comment this out to disable the tooltip :)
fig.canvas.mpl_connect("motion_notify_event", update_tooltip)

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

window.mainloop()