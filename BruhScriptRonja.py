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

def play_closest_sound(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        if isinstance(x, np.ndarray):
            x = x[0]
        if isinstance(y, np.ndarray):
            y = y[0]

        x = float(x)
        y = float(y)

        clicked_point = np.array([x, y])
        selected_index = np.argmin([euclidean_distance(clicked_point, point) for point in pca_results])
        selected_filename = features_dataframe.iloc[selected_index]['Filename']
        selected_file_path = os.path.join(drumFolder, selected_filename)
        y_audio, sr_audio = librosa.load(selected_file_path, sr=None)
        sd.play(y_audio, sr_audio)
        sd.wait()
        coords_label.config(text=f"Clicked coordinates: {x:.2f}, {y:.2f}")
        filename_label.config(text=f"Sound file: {selected_filename}")

        history_text.config(state=tk.NORMAL)
        if len(history) >= MAX_HISTORY:
            history.pop(-1)

        history.insert(0, (x, y, selected_filename))
        update_history_text()

        #history.append((x, y, selected_filename))
        #history_text.delete('1.0', tk.END)
        #for entry in history:
        #    history_text.insert(tk.END, f"Clicked coordinates: {entry[0]:.2f}, {entry[1]:.2f}\n")
        #    history_text.insert(tk.END, f"Sound file: {entry[2]}\n")
        #    history_text.insert(tk.END, "-" * 35 + "\n")
        #history_text.config(state=tk.DISABLED)

def update_history_text():
    history_text.config(state=tk.NORMAL)
    history_text.delete('1.0', tk.END)
    for entry in history:
        history_text.insert(tk.END, f"Clicked coordinates: {entry[0]:.2f}, {entry[1]:.2f}\n")
        history_text.insert(tk.END, f"Sound file: {entry[2]}\n")
        history_text.insert(tk.END, "-" * 35 + "\n")
    history_text.config(state=tk.DISABLED)

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
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        rms_energy = librosa.feature.rms(y=y).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).mean()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr).mean()

        # Store the features
        features_list.append({
            'Filename': filename,
            'Spectral Rolloff': spectral_rolloff,
            'Zero Crossing Rate': zero_crossing_rate,
            'RMS Energy': rms_energy,
            'Spectral Contrast': spectral_contrast,
            'Onset Envelope': onset_env
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
#window.geometry('800x600')
window.title("PCA Results")

plot_frame = tk.Frame(window)
plot_frame.pack(side=tk.LEFT, padx=10, pady=10)

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
ax.set_title('PCA Results on Spread-Out Log-Transformed Features')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
plt.xlim(-2, 2)
plt.ylim(-2, 2)

#left_frame = tk.Frame(window)
#left_frame.pack(padx=10, pady=5, side=tk.LEFT)

#bottom_frame = tk.Frame(window)
#bottom_frame.pack(padx=10, pady=5, side=tk.BOTTOM)

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack()

right_frame = tk.Frame(window)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

coords_label = tk.Label(right_frame, text="", justify='center')
coords_label.pack()

filename_label = tk.Label(right_frame, text="", wraplength=300, justify='center')
filename_label.pack()

empty_label = tk.Label(right_frame, text="")
empty_label.pack()

history_label = tk.Label(right_frame, text="History")
history_label.pack()

history_text = tk.Text(right_frame, height=25, width=35, state=tk.DISABLED)
history_text.pack()

MAX_HISTORY = 8
history = []

#bottom_frame = tk.Frame(window)
#bottom_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

fig.canvas.mpl_connect('button_press_event', play_closest_sound)

window.mainloop()

#plt.show()

# Accessing the loadings
loadings = pca.components_
print("Loadings for the first principal component:", loadings[0])
print("Loadings for the second principal component:", loadings[1])