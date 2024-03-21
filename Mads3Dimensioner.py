import os
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd

def euclidean_distance(point1, point2):
    """Calculates distance between two points."""
    return np.linalg.norm(point1 - point2)

def play_sound(selected_point, scaled_features):
    """Plays the sound closest to the selected point."""
    # Use only the first two dimensions (x, y) for comparison
    scaled_features_array_xy = scaled_features[:, :2]
    selected_index = np.argmin([euclidean_distance(selected_point, point) for point in scaled_features_array_xy])
    selected_filename = features_dataframe.iloc[selected_index]['Filename']
    selected_file_path = os.path.join(drumFolder, selected_filename)
    y, sr = librosa.load(selected_file_path, sr=None)
    sd.stop()  # Stop any currently playing sounds
    sd.play(y, sr)
    sd.wait()

drumFolder = '500_Sounds'
features_list = []

# Ensure the drumFolder exists
if not os.path.exists(drumFolder):
    print(f"The folder '{drumFolder}' does not exist.")
    exit()

for filename in os.listdir(drumFolder):
    if filename.endswith('.wav'):
        file_path = os.path.join(drumFolder, filename)
        y, sr = librosa.load(file_path)
        rms_energy = librosa.feature.rms(y=y).mean()
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        features_list.append({
            'Filename': filename,
            'RMS Energy': rms_energy,
            'Onset Strength': onset_strength,
            'Spectral Rolloff': spectral_rolloff,
        })

features_dataframe = pd.DataFrame(features_list)
features_log_transformed = features_dataframe.drop(columns=['Filename']).apply(np.log)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_log_transformed)

window = tk.Tk()
window.title("Feature Scatter Plot")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(scaled_features[:, 0], scaled_features[:, 1], c=scaled_features[:, 2], cmap='viridis', alpha=0.5, picker=True)

cbar = plt.colorbar(scatter)
cbar.set_label('Scaled Spectral Rolloff')

ax.set_title('Scatter Plot of Scaled Features')
ax.set_xlabel('Scaled RMS Energy')
ax.set_ylabel('Scaled Onset Strength')

tooltip = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"))
tooltip.set_visible(False)

def on_pick(event):
    ind = event.ind[0]
    selected_point = scaled_features[ind, :2]
    play_sound(selected_point, scaled_features)

def update_tooltip(event):
    if tooltip.get_visible():
        tooltip.set_visible(False)
        fig.canvas.draw_idle()
    contains, ind = scatter.contains(event)
    if contains:
        index = ind['ind'][0]
        x, y = scatter.get_offsets()[index]
        tooltip.xy = (x, y)
        tooltip.set_text(features_dataframe['Filename'].iloc[index])
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect("motion_notify_event", update_tooltip)

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

window.mainloop()
