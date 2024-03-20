import os
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
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
    selected_index = np.argmin([euclidean_distance(selected_point, point) for point in pca_results])
    selected_filename = features_dataframe.iloc[selected_index]['Filename']
    selected_file_path = os.path.join(drumFolder, selected_filename)
    y, sr = librosa.load(selected_file_path, sr=None)
    sd.play(y, sr)
    sd.wait()

# Assuming the drumFolder is correctly set to where your .wav files are located
drumFolder = '500_Sounds'

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
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        rms_energy = librosa.feature.rms(y=y).mean()

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
features_dataframe = pd.DataFrame(features_list)

# Before scaling, apply log to the features to ensure they're all positive
# Exclude the filename column for the logarithm transformation
features_log_transformed = features_dataframe.iloc[:, 1:].apply(np.log)

# Scaling the spread-out log-transformed features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_log_transformed)

# Applying PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_features)

# Identify important axes (features) for PCA by looking at the explained variance ratio
print(f"Explained variance by component: {pca.explained_variance_ratio_}")

# Create window
window = tk.Tk()
window.title("PCA Results")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
ax.set_title('PCA Results on Spread-Out Log-Transformed Features')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')

mplcursors.cursor(scatter).connect("add", lambda sel: play_sound(sel.target))

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

window.mainloop()

#plt.show()

# Accessing the loadings
loadings = pca.components_
print("Loadings for the first principal component:", loadings[0])
print("Loadings for the second principal component:", loadings[1])
