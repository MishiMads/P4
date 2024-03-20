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

# GUI
window = tk.Tk()
window.title("You're mom")

plot_frame = ttk.Frame(window)
plot_frame.pack(padx=10, pady=10)

number_of_features = 5


def extract_features(audio):
    y, sr = librosa.load(audio)
    s_features = []

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    rms_energy = librosa.feature.rms(y=y).mean()

    s_features.append(spectral_centroid)
    s_features.append(spectral_bandwidth)
    s_features.append(spectral_rolloff)
    s_features.append(zero_crossing_rate)
    s_features.append(rms_energy)

    return s_features


sound_dir = '500_Sounds'

X = []
sound_files = os.listdir(sound_dir)[:500]
for file in sound_files:
    audio_file = os.path.join(sound_dir, file)
    features = extract_features(audio_file)
    X.append(features)

X = np.array(X)
X_log = np.log(X)
X_sqrt = np.sqrt(X)
X_cbrt = np.cbrt(X)

fig, axs = plt.subplots(number_of_features, 1, figsize=(10, 8), sharex='all')

for i in range(number_of_features):
    axs[i].hist(X_log[:, i], bins=20, color='blue', alpha=0.7)
    axs[i].set_title(f'Feature {i + 1}')
    axs[i].set_ylabel('Frequency')

axs[number_of_features - 1].set_xlabel('Feature Values')

pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_log)

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5, picker=5)
#ax2.title('PCA Log() features')
#ax2.xlabel('PCA Component 1')
#ax2.ylabel('PCA Component 2')

#plt.xlim(-1, 1)
#plt.ylim(-0.2, 0.2)


def on_click(event):
    if event.inaxes == ax2:
        ind = event.ind[0]
        print("BruH!", ind)


fig.canvas.mpl_connect('pick_event', on_click)

canvas = FigureCanvasTkAgg(fig2, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

plt.tight_layout()
plt.show()

window.mainloop()

