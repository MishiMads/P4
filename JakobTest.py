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


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def play_closest_sound(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        clicked_point = np.array([x, y])
        selected_index = np.argmin([euclidean_distance(clicked_point, point) for point in feature_list])
        selected_filename = features_dataframe.iloc[selected_index]['Filename']
        selected_file_path = os.path.join(sound_dir, selected_filename)
        y, sr = librosa.load(selected_file_path, sr=None)
        sd.play(y, sr)
        sd.wait()
        coords.config(text=f"Clicked coordinates: {x:.2f}, {y:.2f}")
        filename_label.config(text="Sound file: {selected_filename")


window = tk.Tk()
window.title("PCA Results")


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


left_frame = tk.Frame(window)
left_frame.pack(padx=10, pady=5, side=tk.LEFT)

bottom_frame = tk.Frame(window)
bottom_frame.pack(padx=10, pady=5, side=tk.BOTTOM, fill=tk.X)

canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

coords = tk.Label(bottom_frame, text="")
coords.pack(side=tk.RIGHT)

filename_label = tk.Label(bottom_frame, text="", wraplength=300)
filename_label.pack(side=tk.LEFT)

fig.canvas.mpl_connect('button_press_event', play_closest_sound)

window.mainloop()

loadings = pca.components_
print("Loadings for the first principal component:", loadings[0])
print("Loadings for the second principal component:", loadings[1])

plt.show()
