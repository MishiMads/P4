import librosa
import os
import librosa.feature
import matplotlib.pyplot as plt
import pandas


class SoundFile:
    def __init__(self, path, filename, bright, warm, boomy, tight, punchy, sharp, muddy, crisp, resonant, metallic):
        self.path = path
        self.filename = filename
        self.bright = bright
        self.warm = warm
        self.boomy = boomy
        self.tight = tight
        self.punchy = punchy
        self.sharp = sharp
        self.muddy = muddy
        self.crisp = crisp
        self.resonant = resonant
        self.metallic = metallic


sound1 = SoundFile('/JakobLyde/004400-rdd_kick718.wav', "sound1", False, False, True, True, True, False, False, False, False, False)
sound2 = SoundFile('/JakobLyde/004401-rdd_kick719.wav', "sound2", False, False, False, True, False, False, False, False, False, False)
sound3 = SoundFile('/JakobLyde/004403-rdd_kick720.wav', "sound3", False, False, False, True, True, False, False, False, False, False)
sound4 = SoundFile('/JakobLyde/004404-rdd_kick721.wav', "sound4", False, False, True, True, True, False, True, False, False, False)
sound5 = SoundFile('/JakobLyde/004405-rdd_kick722.wav', "sound5", False, False, True, True, True, True, True, False, False, False)
sound6 = SoundFile('/JakobLyde/004406-rdd_kick723.wav', "sound6", False, False, True, False, True, False, True, False, True, False)
sound7 = SoundFile('/JakobLyde/004407-rdd_kick724.wav', "sound7", False, True, True, False, True, False, False, False, False, False)
sound8 = SoundFile('/JakobLyde/004408-rdd_kick725.wav', "sound8", False, False, True, False, True, False, True, False, True, False)
sound9 = SoundFile('/JakobLyde/004409-rdd_kick726.wav', "sound9", False, False, True, True, True, True, False, False, False, False)
sound10 = SoundFile('/JakobLyde/004410-rdd_kick727.wav', "sound10", False, False, False, True, True, False, False, False, False, False)
sound11 = SoundFile('/JakobLyde/004411-rdd_kick728.wav', "sound11", False, False, True, False, False, False, False, False, True, False)

directory = r'C:\Users\jakob\Desktop\JakobLytteLyde'

bruhListe = [sound1, sound2, sound3, sound4, sound5, sound6, sound7, sound8, sound9, sound10, sound11]


def plot_sound_characteristics(sound_files, feature1):
    plt.figure(figsize=(8, 6))
    for sound in sound_files:
        y, sr = librosa.load(sound.path)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

        plt.scatter(getattr(sound, feature1), spectral_centroid, label=sound.filename)
    plt.xlabel(feature1)
    plt.ylabel("Specral Centroid")
    plt.title("Comparison of Sound Characteristics")
    plt.legend()
    plt.grid(True)
    plt.show()



plot_sound_characteristics(bruhListe, 'tight')
