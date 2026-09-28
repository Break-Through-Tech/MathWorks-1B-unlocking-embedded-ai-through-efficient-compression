import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io, signal, stats
from sklearn.decomposition import PCA

DATA_DIR = 'data'
OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

fs = 48828

shaft = 25
bpfo = 8 / 2 * shaft * (1 - 0.235 / 1.245)
bpfi = 8 / 2 * shaft * (1 + 0.235 / 1.245)

classes = ['Normal', 'OuterRaceFault', 'InnerRaceFault']
names = {'Normal': 'Normal', 'OuterRaceFault': 'Outer Race Fault', 'InnerRaceFault': 'Inner Race Fault'}
colors = {'Normal': 'C0', 'OuterRaceFault': 'C1', 'InnerRaceFault': 'C2'}

train = io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
test = io.loadmat(os.path.join(DATA_DIR, 'test.mat'))
X = train['trainData']
y = np.array([label[0] for label in train['trainLabels'].flatten()])
X_test = test['testData']

example = {}
for c in classes:
    idx = np.where(y == c)[0]
    k = stats.kurtosis(X[idx], axis=1)
    example[c] = idx[np.argsort(k)[len(k) // 2]]


def envelope_spectrum(data):
    env = np.abs(signal.hilbert(data, axis=1))
    env = env - env.mean(axis=1, keepdims=True)
    window = np.hanning(data.shape[1])
    nfft = 2 ** 16
    spec = np.abs(np.fft.rfft(env * window, n=nfft, axis=1)) / (window.sum() / 2)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    return freqs, spec

# median power spectral density by class
f, P = signal.welch(X, fs=fs, nperseg=1024, axis=1)

plt.figure(figsize=(9, 5))
for c in classes:
    plt.semilogy(f / 1000, np.median(P[y == c], axis=0), color=colors[c], label=names[c])
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power Spectral Density')
plt.title('Median Power Spectral Density by Class')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_power_spectral_density.png'), dpi=150)
plt.close()

# spectograms
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, c in zip(axes, classes):
    fspec, tspec, S = signal.spectrogram(X[example[c]], fs=fs, nperseg=256, noverlap=224)
    im = ax.pcolormesh(tspec * 1000, fspec / 1000, 10 * np.log10(S), vmin=-80, vmax=-20, shading='auto')
    ax.set_title(names[c])
    ax.set_xlabel('Time (ms)')
axes[0].set_ylabel('Frequency (kHz)')
fig.colorbar(im, ax=axes, label='Power (dB)')
fig.suptitle('Spectrograms by Class', y=1.03)
plt.savefig(os.path.join(OUT_DIR, '03_spectrograms.png'), dpi=150, bbox_inches='tight')
plt.close()

# mean envelope spectrum with BPFO/BPFI harmonics
fe, E = envelope_spectrum(X)
keep = fe <= 420

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
for ax, c in zip(axes, classes):
    ax.plot(fe[keep], E[y == c][:, keep].mean(axis=0), color=colors[c])
    ax.set_title(names[c])
    ax.set_ylabel('Amplitude')

for n in range(1, 5):
    axes[1].axvline(n * bpfo, color='k', linestyle='--', linewidth=1, label='BPFO harmonics (81.1 Hz)' if n == 1 else None)
for n in range(1, 4):
    axes[2].axvline(n * bpfi, color='k', linestyle='--', linewidth=1, label='BPFI harmonics (118.9 Hz)' if n == 1 else None)
axes[1].legend(loc='upper right')
axes[2].legend(loc='upper right')
axes[2].set_xlabel('Frequency (Hz)')
fig.suptitle('Mean Envelope Spectrum by Class')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_envelope_spectrum.png'), dpi=150)
plt.close()

# simple PCA
_, P_train = signal.welch(X, fs=fs, nperseg=512, axis=1)
_, P_test = signal.welch(X_test, fs=fs, nperseg=512, axis=1)
log_train = np.log10(P_train)
log_test = np.log10(P_test)
mean_spec = log_train.mean(axis=0)

pca = PCA(n_components=2)
Z = pca.fit_transform(log_train - mean_spec)
Z_test = pca.transform(log_test - mean_spec)

plt.figure(figsize=(8, 6))
for c in classes:
    plt.scatter(Z[y == c, 0], Z[y == c, 1], s=15, color=colors[c], alpha=0.5, label=names[c] + ' (train)')
plt.scatter(Z_test[:, 0], Z_test[:, 1], s=25, color='k', marker='x', label='Test')
plt.xlabel('PC 1 (%.0f%%)' % (pca.explained_variance_ratio_[0] * 100))
plt.ylabel('PC 2 (%.0f%%)' % (pca.explained_variance_ratio_[1] * 100))
plt.title('PCA of Log Power Spectra')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_pca.png'), dpi=150)
plt.close()

print('Saved 4 figures to', OUT_DIR)
