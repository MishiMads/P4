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


sound1 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004400-rdd_kick718.wav', "sound1", False, False, True, True, True, False, False, False, False, False)
sound2 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004401-rdd_kick719.wav', "sound2", False, False, False, True, False, False, False, False, False, False)
sound3 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004403-rdd_kick720.wav', "sound3", False, False, False, True, True, False, False, False, False, False)
sound4 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004404-rdd_kick721.wav', "sound4", False, False, True, True, True, False, True, False, False, False)
sound5 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004405-rdd_kick722.wav', "sound5", False, False, True, True, True, True, True, False, False, False)
sound6 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004406-rdd_kick723.wav', "sound6", False, False, True, False, True, False, True, False, True, False)
sound7 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004407-rdd_kick724.wav', "sound7", False, True, True, False, True, False, False, False, False, False)
sound8 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004408-rdd_kick725.wav', "sound8", False, False, True, False, True, False, True, False, True, False)
sound9 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004409-rdd_kick726.wav', "sound9", False, False, True, True, True, True, False, False, False, False)
sound10 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004410-rdd_kick727.wav', "sound10", False, False, False, True, True, False, False, False, False, False)
sound11 = SoundFile(r'C:\Users\jakob\Desktop\JakobLytteLyde\004411-rdd_kick728.wav', "sound11", False, False, True, False, False, False, False, False, True, False)

directory = r'C:\Users\jakob\Desktop\JakobLytteLyde'

bruhListe = []
