from scipy.io import wavfile as wf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal


COLORS = ['r', 'b', 'g', 'y', 'purple', 'brown', 'pink']


def plot_times_and_frequencies(files):
    rates = []
    datas = []
    filenames = []
    for file in files:
        rate, data = wf.read(file)
        left_channel = data.T[0]
        rates.append(rate)
        datas.append(left_channel)
        filenames.append(file)

    #plot in time domain
    for i in range(len(datas)):
        length = datas[i].shape[0] / rates[i]
        time = np.linspace(0., length, datas[i].shape[0])
        plt.plot(time, datas[i], label=filenames[i], color=COLORS[i])
    plt.title('Amplitudes vs. time')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [v]")
    plt.legend()
    plt.show()



    #plot in frequency domain
    for i in range(len(datas)):
        normalized = [(ele / 2 ** 8.) * 2 - 1 for ele in datas[i]]
        fouriered = fft(normalized)
        half = len(fouriered) // 2
        plt.plot(abs(fouriered[:(half - 1)]), label=filenames[i], color=COLORS[i])
    plt.title('Amplitudes vs. frequency of musical notes.')
    plt.legend()
    plt.xlabel("Frequency [1/s]")
    plt.xlim([0, 400])
    plt.ylabel("Amplitude [v]")
    plt.show()

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T

def plot_stft(filename):
    rate, data = wf.read(filename)  # reads wav file.
    left_channel = data.T[0]

    # plot amplitude vs time:
    length = data.shape[0] / rate
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, left_channel, label="Left channel")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [v]")
    plt.title('Amplitude vs. time')
    plt.show()

    # plot in frequency domain, with stft.
    normalized = [(ele / 2 ** 8.) * 2 - 1 for ele in left_channel]
    fouriered = stft(normalized)
    # fouriered = fft(normalized)
    half = len(fouriered) // 2
    plt.plot(np.abs(fouriered[:(half - 1)]), 'r')
    plt.xlabel("Frequency [1/s]")
    # plt.Axes.set_xlim(right=1000)
    plt.ylabel("Amplitude [v]")
    plt.title('Amplitude vs. frequency by STFT')
    plt.show()



if __name__ == "__main__":
    notes = ['do.wav', 're.wav', 'mi.wav', 'fa.wav', 'sol.wav', 'la.wav', 'si.wav']
    plot_times_and_frequencies(notes)

    # plot_stft('jony-line.wav')
