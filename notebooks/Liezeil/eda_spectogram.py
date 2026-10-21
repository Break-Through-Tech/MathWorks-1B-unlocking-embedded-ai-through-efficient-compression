# Plots spectrograms to explore how the classes differ
from scipy.signal import spectrogram
from pathlib import Path
from spectro import make_spec
import numpy as np
import matplotlib.pyplot as plt
from spectro import make_all

# Settings
DATA_FILE = Path(__file__).parent / "standardized_global.npz"
CLASS_NAMES = ["Normal", "InnerRaceFault", "OuterRaceFault"]

FS = 48828   # sample rate in Hz
N_FFT = 256  # window length in samples
HOP = 64     # how far the window moves each step
HERE = Path(__file__).resolve().parent

data = np.load(DATA_FILE)
X_train = data["X_train"]
y_train = data["y_train"]

N_FFTS = (128, 256, 512)
PAIR = (0, 2)  

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for r, c in enumerate(PAIR):
    sigs = X_train[y_train == c]
    for k, n in enumerate(N_FFTS):
        mean_spec = np.stack([make_spec(x, n, HOP) for x in sigs]).mean(axis=0)
        axes[r, k].imshow(mean_spec, aspect="auto", origin="lower")
        axes[r, k].set_title(f"{CLASS_NAMES[c]}, n_fft={n}")
plt.show()

N_FFT = 128
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for r, c in enumerate(PAIR):
    sigs = X_train[y_train == c]
    for k in range(4):
        axes[r, k].imshow(make_spec(sigs[k], N_FFT, HOP), aspect="auto", origin="lower")
        axes[r, k].set_title(f"{CLASS_NAMES[c]} #{k}")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for c, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
    avg = make_all(X_train[y_train == c], 128, 64).mean(axis=0)
    ax.imshow(avg, aspect="auto", origin="lower")
    ax.set_title(name)
    ax.set_xlabel("time window")
axes[0].set_ylabel("frequency row")
plt.tight_layout()
plt.savefig(HERE / "spec_slide.png", dpi=200)