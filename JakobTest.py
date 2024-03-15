import librosa
import scipy.fft
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

directory = '500_Sounds'


# Show waveform of a sound file
def plot_waveform(file):
    data, samplerate = sf.read(file)

    if len(data.shape) > 1:
        data = data[:, 0]

    duration = len(data) / samplerate
    time = np.linspace(0., duration, len(data))

    plt.figure(figsize=(10, 4))
    plt.plot(time, data, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

    fft_output = np.fft.fft(data)
    fft_magnitude = np.abs(fft_output)
    frequencies = np.fft.fftfreq(len(data), 1 / samplerate)

    plt.figure(figsize=(10, 4))
    plt.plot(frequencies[:len(data)//2], fft_magnitude[:len(data)//2], color='blue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT')
    plt.xlim(0, 2000)
    plt.grid()
    plt.show()


filename = '500_Sounds/test1.wav'
plot_waveform(filename)
