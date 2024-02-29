import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



class SoundFile:
    def __init__(self, path, filename, **kwargs):
        self.path = path
        self.filename = filename
        for key, value in kwargs.items():
            setattr(self, key, value)


sound1 = SoundFile(path=r'JakobLyde/004400-rdd_kick718.wav', filename="sound1", bright=False, warm=False, boomy=True, tight=True, punchy=True, sharp=False, muddy=False, crisp=False, resonant=False, metallic=False)
sound2 = SoundFile(path=r'JakobLyde/004401-rdd_kick719.wav', filename="sound2", bright=False, warm=False, boomy=False, tight=True, punchy=False, sharp=False, muddy=False, crisp=False, resonant=False, metallic=False)
sound3 = SoundFile(path=r'JakobLyde/004403-rdd_kick720.wav', filename="sound3", bright=False, warm=False, boomy=False, tight=True, punchy=True, sharp=False, muddy=False, crisp=False, resonant=False, metallic=False)
sound4 = SoundFile(path=r'JakobLyde/004404-rdd_kick721.wav', filename="sound4", bright=False, warm=False, boomy=True, tight=True, punchy=True, sharp=False, muddy=True, crisp=False, resonant=False, metallic=False)
sound5 = SoundFile(path=r'JakobLyde/004405-rdd_kick722.wav', filename="sound5", bright=False, warm=False, boomy=True, tight=True, punchy=True, sharp=True, muddy=True, crisp=False, resonant=False, metallic=False)
sound6 = SoundFile(path=r'JakobLyde/004406-rdd_kick723.wav', filename="sound6", bright=False, warm=False, boomy=True, tight=False, punchy=True, sharp=False, muddy=True, crisp=False, resonant=True, metallic=False)
sound7 = SoundFile(path=r'JakobLyde/004407-rdd_kick724.wav', filename="sound7", bright=False, warm=True, boomy=True, tight=False, punchy=True, sharp=False, muddy=False, crisp=False, resonant=False, metallic=False)
sound8 = SoundFile(path=r'JakobLyde/004408-rdd_kick725.wav', filename="sound8", bright=False, warm=False, boomy=True, tight=False, punchy=True, sharp=False, muddy=True, crisp=False, resonant=True, metallic=False)
sound9 = SoundFile(path=r'JakobLyde/004409-rdd_kick726.wav', filename="sound9", bright=False, warm=False, boomy=True, tight=True, punchy=True, sharp=True, muddy=False, crisp=False, resonant=False, metallic=False)
sound10 = SoundFile(path=r'JakobLyde/004410-rdd_kick727.wav', filename="sound10", bright=False, warm=False, boomy=False, tight=True, punchy=True, sharp=False, muddy=False, crisp=False, resonant=False, metallic=False)
sound11 = SoundFile(path=r'JakobLyde/004411-rdd_kick728.wav', filename="sound11", bright=False, warm=False, boomy=True, tight=False, punchy=False, sharp=False, muddy=False, crisp=False, resonant=True, metallic=False)



soundFileList = [sound1, sound2, sound3, sound4, sound5, sound6, sound7, sound8, sound9, sound10, sound11]

# Here we extract the features from the sound files and store them in a list of dictionaries called features_list
features_list = []
for sound in soundFileList:
    y, sr = librosa.load(sound.path)
    features = {
        'filename': sound.filename,
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).mean(),
        'rms_energy': librosa.feature.rms(y=y).mean(),
    }
    # Here we also include the the qualitative attributes
    for attr in ['bright', 'warm', 'boomy', 'tight', 'punchy', 'sharp', 'muddy', 'crisp', 'resonant', 'metallic']:
        features[attr] = getattr(sound, attr)
    features_list.append(features)

df = pd.DataFrame(features_list)
X = df[['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate', 'rms_energy']]
X_standardized = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_standardized)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])


# Here we define the plotting function
def plot_pca_by_attribute(df, principal_df, attribute):
    final_df = pd.concat([principal_df, df[['filename', attribute]]], axis=1)
    colors = ['blue' if not x else 'orange' for x in final_df[attribute]]
    plt.figure(figsize=(10, 7))
    plt.scatter(final_df['PC1'], final_df['PC2'], c=colors)
    for i, txt in enumerate(final_df['filename']):
        plt.annotate(txt, (final_df['PC1'].iloc[i], final_df['PC2'].iloc[i]))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA of Sound Characteristics: {attribute.capitalize()}')
    plt.grid(True)
    plt.show()

# Here we call the function for each qualitative attribute
attributes = ['bright', 'warm', 'boomy', 'tight', 'punchy', 'sharp', 'muddy', 'crisp', 'resonant', 'metallic']
for attribute in attributes:
    plot_pca_by_attribute(df, principal_df, attribute)