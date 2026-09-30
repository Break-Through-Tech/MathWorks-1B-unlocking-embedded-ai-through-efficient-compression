from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Resolve paths relative to this script.
PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the training data.
data = loadmat(PROJECT_DIR / "data" / "train.mat")
signals = data["trainData"]
labels = np.array([
    str(label.item()) for label in data["trainLabels"].ravel()
])
classes, counts = np.unique(labels, return_counts=True)

for class_name, count in zip(classes, counts):
    print(f"{class_name}: {count} samples")

print(f"Total: {len(labels)} samples")
print("Signal matrix:", signals.shape)
print("Class counts:", np.unique(labels, return_counts=True))

# Convert sample positions into time in milliseconds.
sample_rate = 48_828
time_ms = np.arange(signals.shape[1]) / sample_rate * 1000

classes = ["Normal", "InnerRaceFault", "OuterRaceFault"]
# Set a class to (lower, upper) to choose manual amplitude limits.
# None fits the selected signal in that class, with 10% padding.
vertical_limits = {
    "Normal": None,
    "InnerRaceFault": None,
    "OuterRaceFault": None,
}
# A fixed seed keeps the random examples reproducible; change it for a new set.
rng = np.random.default_rng(42)
fig, axes = plt.subplots(
    len(classes), 1,
    figsize=(12, 9),
    sharex=True,
    sharey=False,
)

for ax, class_name in zip(axes, classes):
    # Randomly select one training segment belonging to this class.
    candidates = np.flatnonzero(labels == class_name)
    index = rng.choice(candidates)

    ax.plot(time_ms, signals[index], linewidth=0.7)
    limits = vertical_limits[class_name]
    # Fit the selected signal's largest peak with 10% padding.
    if limits is None:
        peak = float(np.max(np.abs(signals[index])))
        limit = peak * 1.1 if peak > 0 else 1.0
        limits = (-limit, limit)
    ax.set_ylim(*limits)
    ax.set_title(class_name)
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Time within each segment (ms)")
fig.suptitle("Bearing vibration signals — one random example per class")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "example_signals.png", dpi=200)

# Summarize every training segment, scaling each class independently.
fig_spectra, axes_spectra = plt.subplots(
    len(classes), 1, figsize=(12, 9), sharex=True, sharey=False,
)
sample_count = signals.shape[1]
frequencies = np.fft.rfftfreq(sample_count, d=1 / sample_rate)
window = np.hanning(sample_count)

for ax, class_name in zip(axes_spectra, classes):
    class_signals = signals[labels == class_name]
    # Compute each segment's spectrum separately; do not average raw waveforms.
    centered = class_signals - class_signals.mean(axis=1, keepdims=True)
    amplitude = np.abs(np.fft.rfft(centered * window, axis=1)) / window.sum()
    # Convert to a one-sided amplitude spectrum; keep DC and Nyquist undoubled.
    if sample_count % 2 == 0:
        amplitude[:, 1:-1] *= 2
    else:
        amplitude[:, 1:] *= 2

    # At each frequency, show the median and middle 50% across segments.
    lower, median, upper = np.percentile(amplitude, [25, 50, 75], axis=0)
    ax.fill_between(frequencies / 1000, lower, upper, alpha=0.25,
                    label="Middle 50% of segments")
    ax.plot(frequencies / 1000, median, linewidth=0.9, label="Median")
    ax.set_title(f"{class_name} (n={len(class_signals)})")
    ax.set_ylabel("Spectral amplitude")
    # Fit the full shaded band for this class, with 10% padding.
    spectrum_max = float(upper.max())
    ax.set_ylim(0, spectrum_max * 1.1 if spectrum_max > 0 else 1.0)
    ax.set_xlim(0, sample_rate / 2000)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

axes_spectra[-1].set_xlabel("Frequency (kHz)")
fig_spectra.suptitle("Frequency spectra across all training segments")
fig_spectra.tight_layout()
fig_spectra.savefig(OUTPUT_DIR / "frequency_spectra.png", dpi=200)

# RMS measures signal strength; axis=1 calculates one value per segment.
rms = np.sqrt(np.mean(signals**2, axis=1))

# Peak-to-peak measures each segment's amplitude range (maximum - minimum).
peak_to_peak = np.ptp(signals, axis=1)

fig_summary, axes_summary = plt.subplots(1, 2, figsize=(12, 5))
metrics = [
    ("RMS amplitude", rms),
    ("Peak-to-peak amplitude", peak_to_peak),
]
colors = ["#4C78A8", "#F58518", "#54A24B"]

for ax, (metric_name, values) in zip(axes_summary, metrics):
    # Group all segment values by class for the box plots.
    grouped_values = [values[labels == name] for name in classes]
    boxes = ax.boxplot(grouped_values, patch_artist=True, showfliers=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.65)

    # Boxes show the middle 50% and median; points mark values beyond the whiskers.
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels([
        f"{name}\n(n={len(group)})"
        for name, group in zip(classes, grouped_values)
    ])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name + " (original signal units)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)

fig_summary.suptitle("Amplitude summaries across all training segments")
fig_summary.tight_layout()
fig_summary.savefig(OUTPUT_DIR / "amplitude_boxplots.png", dpi=200)

# Display the waveforms, frequency spectra, and summary plots.
plt.show()
