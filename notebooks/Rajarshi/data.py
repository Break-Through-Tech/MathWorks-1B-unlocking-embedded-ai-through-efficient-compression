from scipy import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/train.mat'
test = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/test.mat'
val = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/val.mat'

train_data = io.loadmat(train)
test_data = io.loadmat(test)
val_data = io.loadmat(val)

print(train_data.keys())

my_matrix = train_data['trainData']
print(my_matrix)

class_order = ['Normal', 'OuterRaceFault', 'InnerRaceFault']


def plot_class_distribution(mat_data, labels_key, title, out_path):
    labels = pd.Series([row[0][0] for row in mat_data[labels_key]])
    counts = labels.value_counts().reindex(class_order)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(class_order, counts.values, color='#2a78d6', width=0.6)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                 str(int(count)), ha='center', va='bottom', color='#0b0b0b', fontsize=10)

    ax.set_title(title, color='#0b0b0b', fontsize=13, pad=12)
    ax.set_ylabel('Number of instances', color='#52514e')
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e')
    ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


plot_class_distribution(
    train_data, 'trainLabels', 'Training Set Class Distribution',
    'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/train_class_distribution.png')
plot_class_distribution(
    test_data, 'testLabels', 'Test Set Class Distribution',
    'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/test_class_distribution.png')
plot_class_distribution(
    val_data, 'valLabels', 'Validation Set Class Distribution',
    'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/val_class_distribution.png')

# Signal amplitude distribution per class, training set only
class_colors = {'Normal': '#2a78d6', 'OuterRaceFault': '#eb6834', 'InnerRaceFault': '#1baf7a'}
train_labels = np.array([row[0][0] for row in train_data['trainLabels']])
signal_by_class = {cls: train_data['trainData'][train_labels == cls].ravel() for cls in class_order}

# Boxplot: compares spread/outliers across classes on one shared axis
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([signal_by_class[cls] for cls in class_order], labels=class_order, patch_artist=True,
                 medianprops=dict(color='#0b0b0b', linewidth=1.5),
                 flierprops=dict(markersize=2, alpha=0.25, markeredgecolor='none'))
for patch, cls in zip(bp['boxes'], class_order):
    patch.set_facecolor(class_colors[cls])
    patch.set_alpha(0.55)
    patch.set_edgecolor(class_colors[cls])
for element in ('whiskers', 'caps'):
    for line in bp[element]:
        line.set_color('#898781')
for flier, cls in zip(bp['fliers'], class_order):
    flier.set_markerfacecolor(class_colors[cls])

all_values = np.concatenate(list(signal_by_class.values()))
lo, hi = np.percentile(all_values, [0.5, 99.5])
pad = (hi - lo) * 0.1
ax.set_ylim(lo - pad, hi + pad)

ax.set_title('Training Signal Amplitude by Class', color='#0b0b0b', fontsize=13, pad=12)
ax.set_ylabel('Signal amplitude', color='#52514e')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#c3c2b7')
ax.spines['bottom'].set_color('#c3c2b7')
ax.tick_params(colors='#52514e')
ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8)
ax.set_axisbelow(True)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')
fig.tight_layout()
plt.savefig('MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/train_signal_boxplot.png', dpi=150)
plt.show()

# Histogram small multiples: each class keeps its own x-range since scales differ sharply
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, cls in zip(axes, class_order):
    values = signal_by_class[cls]
    ax.hist(values, bins=80, color=class_colors[cls], alpha=0.75, edgecolor='none')
    ax.set_title(cls, color='#0b0b0b', fontsize=12)
    ax.set_xlabel('Signal amplitude', color='#52514e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e')
    ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor('#fcfcfb')
axes[0].set_ylabel('Count', color='#52514e')
fig.suptitle('Training Signal Amplitude Distribution per Class', color='#0b0b0b', fontsize=13)
fig.patch.set_facecolor('#fcfcfb')
fig.tight_layout()
plt.savefig('MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/train_signal_histograms.png', dpi=150)
plt.show()

# One example waveform per class, plotted over time (own y-scale per class since amplitudes differ a lot)
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for ax, cls in zip(axes, class_order):
    example = train_data['trainData'][train_labels == cls][0]
    ax.plot(example, color=class_colors[cls], linewidth=0.8)
    ax.set_ylabel(cls, color='#0b0b0b', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e')
    ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor('#fcfcfb')
axes[-1].set_xlabel('Sample index', color='#52514e')
fig.suptitle('Example Training Waveform per Class', color='#0b0b0b', fontsize=13)
fig.patch.set_facecolor('#fcfcfb')
fig.tight_layout()
plt.savefig('MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/train_example_waveforms.png', dpi=150)
plt.show()

# Z-score of the first training instance
sample = train_data['trainData'][0]
z_score = (sample - sample.mean()) / sample.std()
print(z_score)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(z_score, bins=50, color='#2a78d6', edgecolor='none')
ax.set_title('Z-score Distribution (trainData[0])', color='#0b0b0b', fontsize=13, pad=12)
ax.set_xlabel('Z-score', color='#52514e')
ax.set_ylabel('Count', color='#52514e')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#c3c2b7')
ax.spines['bottom'].set_color('#c3c2b7')
ax.tick_params(colors='#52514e')
ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8)
ax.set_axisbelow(True)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')
fig.tight_layout()
plt.savefig('MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/train_zscore_histogram.png', dpi=150)
plt.show()

# Outliers per class: full-range boxplot (no percentile clipping) so fliers stay visible
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([signal_by_class[cls] for cls in class_order], labels=class_order, patch_artist=True,
                 medianprops=dict(color='#0b0b0b', linewidth=1.5),
                 flierprops=dict(markersize=3, alpha=0.35, markeredgecolor='none'))
for patch, cls in zip(bp['boxes'], class_order):
    patch.set_facecolor(class_colors[cls])
    patch.set_alpha(0.55)
    patch.set_edgecolor(class_colors[cls])
for element in ('whiskers', 'caps'):
    for line in bp[element]:
        line.set_color('#898781')
for flier, cls in zip(bp['fliers'], class_order):
    flier.set_markerfacecolor(class_colors[cls])

ax.set_title('Training Signal Outliers by Class', color='#0b0b0b', fontsize=13, pad=12)
ax.set_ylabel('Signal amplitude', color='#52514e')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#c3c2b7')
ax.spines['bottom'].set_color('#c3c2b7')
ax.tick_params(colors='#52514e')
ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8)
ax.set_axisbelow(True)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')
fig.tight_layout()
plt.savefig('MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/train_outliers_by_class.png', dpi=150)
plt.show()