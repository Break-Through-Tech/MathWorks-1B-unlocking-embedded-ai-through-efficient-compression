import os

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR

SAMPLE_RATE = 48_828
class_order = ['InnerRaceFault', 'Normal', 'OuterRaceFault']

# load data + basic checks
train = scipy.io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
val = scipy.io.loadmat(os.path.join(DATA_DIR, 'val.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_train = train['trainData']       # shape: (393, 5000)
y_train = [label[0] for label in train['trainLabels'].flatten()]
X_val = val['valData']
X_test = test['testData']

print('X_train shape:', X_train.shape)
print('NaN count in X_train:', np.isnan(X_train).sum())

classes_y, counts_y = np.unique(y_train, return_counts=True)
print('Class distribution:', dict(zip(classes_y, counts_y)))

time = np.arange(X_train.shape[1]) / SAMPLE_RATE

# some waveforms
example_indices = [34, 100, 3, 140, 200, 236, 300, 350]
fig, axes = plt.subplots(len(example_indices), 1, figsize=(9, 2 * len(example_indices)), sharex=True)
for ax, idx in zip(axes, example_indices):
    ax.plot(time, X_train[idx], linewidth=1)
    ax.set_ylabel(f'idx {idx}\n({y_train[idx]})', fontsize=8)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('Example Raw Waveforms')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '01_example_waveforms.png'), dpi=150)
plt.close(fig)

# new df with stats for each signal
signal_stats_df = pd.DataFrame({
    'label': y_train,
    'mean': np.mean(X_train, axis=1),
    'median': np.median(X_train, axis=1),
    'std': np.std(X_train, axis=1),
    'min': np.min(X_train, axis=1),
    'max': np.max(X_train, axis=1),
})
print(signal_stats_df.head())

class_stats_df = signal_stats_df.groupby('label', as_index=False).mean(numeric_only=True)
print(class_stats_df)

# sd analysis
plt.figure(figsize=(9, 6))
sns.boxplot(data=signal_stats_df, x='label', y='std', order=class_order)
plt.title('Signal Standard Deviation by Bearing Condition')
plt.xlabel('Bearing Condition')
plt.ylabel('Standard Deviation Across 5,000 Measurements')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_std_boxplot.png'), dpi=150)
plt.close()

# new df with potential features
# rms: avg magnitude of varying quantity
# kurtosis: how heavily points clustered around tails
signal_features = pd.DataFrame({
    'label': y_train,
    'rms': np.sqrt(np.mean(X_train ** 2, axis=1)),   # overall intensity
    'peak_to_peak': np.ptp(X_train, axis=1),
    'max_absolute': np.max(np.abs(X_train), axis=1),
    'skewness': skew(X_train, axis=1),
    'kurtosis': kurtosis(X_train, axis=1),           # peakiness / outlier-proneness
    'sd': np.std(X_train, axis=1),
})
print(signal_features.head())

# rms/kurtosis separates irf, normal and orf overlap more
plt.figure()
sns.scatterplot(data=signal_features, x='rms', y='kurtosis', hue='label')
plt.title('RMS vs. Kurtosis')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_rms_vs_kurtosis.png'), dpi=150)
plt.close()

