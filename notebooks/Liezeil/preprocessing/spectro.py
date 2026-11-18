# Functions that turn signals into spectrograms
import numpy as np
from scipy.signal import spectrogram

FS = 48828  # sample rate in Hz


def make_spec(signal, n_fft, hop):
    # magnitude spectrogram, then log1p to shrink big values
    _, _, sxx = spectrogram(
        signal, fs=FS, nperseg=n_fft, noverlap=n_fft - hop, mode="magnitude"
    )
    return np.log1p(sxx)

def make_all(X, n_fft, hop):
    # one spectrogram per row of X, stacked into (N, rows, cols)
    return np.stack([make_spec(x, n_fft, hop) for x in X])