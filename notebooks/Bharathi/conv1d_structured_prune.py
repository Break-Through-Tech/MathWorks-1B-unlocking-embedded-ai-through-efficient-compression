import os
import copy
import time

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR
BASELINE_CHECKPOINT_PATH = os.path.join(OUT_DIR, 'conv1d_fp32.pt')

BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 5e-4
CALIBRATION_BATCHES = 20
RANDOM_SEED = 42

# Channel configs to try, shrinking (conv1, conv2, conv3) down from the 16/32/64 baseline.
SHRINK_CONFIGS = [
    (8, 16, 32),
    (4, 8, 16),
    (2, 4, 8),
]

torch.manual_seed(RANDOM_SEED)


# ---- Model (must match conv1d.py's architecture) ----
class Conv1DClassifier(nn.Module):
    def __init__(self, conv_channels, kernel_sizes, num_classes):
        super().__init__()
        c1, c2, c3 = conv_channels
        k1, k2, k3 = kernel_sizes

        self.conv1 = nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(c1), nn.BatchNorm1d(c2), nn.BatchNorm1d(c3)
        self.relu1, self.relu2, self.relu3 = nn.ReLU(), nn.ReLU(), nn.ReLU()
        self.pool1, self.pool2 = nn.MaxPool1d(2), nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# ---- Load baseline checkpoint + data ----
checkpoint = torch.load(BASELINE_CHECKPOINT_PATH, map_location='cpu')
class_order = checkpoint['class_order']
label_to_idx = {c: i for i, c in enumerate(class_order)}
train_mean = checkpoint['train_mean']
train_std = checkpoint['train_std']
kernel_sizes = checkpoint['kernel_sizes']

baseline_model = Conv1DClassifier(checkpoint['conv_channels'], kernel_sizes, num_classes=len(class_order))
baseline_model.load_state_dict(checkpoint['state_dict'])
baseline_model.eval()

train = scipy.io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
val = scipy.io.loadmat(os.path.join(DATA_DIR, 'val.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_train = (train['trainData'] - train_mean) / train_std
y_train = [label[0] for label in train['trainLabels'].flatten()]
X_val = (val['valData'] - train_mean) / train_std
y_val = [label[0] for label in val['valLabels'].flatten()]
X_test = (test['testData'] - train_mean) / train_std
y_test = [label[0] for label in test['testLabels'].flatten()]


def to_dataset(X, y):
    X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    y_t = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)
    return TensorDataset(X_t, y_t)


train_loader = DataLoader(to_dataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(to_dataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(to_dataset(X_test, y_test), batch_size=BATCH_SIZE)


def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch).argmax(dim=1).numpy()
            preds_all.extend(preds)
            labels_all.extend(y_batch.numpy())
    return np.array(labels_all), np.array(preds_all)


def accuracy(model, loader):
    y_true, y_pred = evaluate(model, loader)
    return accuracy_score(y_true, y_pred)


def fine_tune(model, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    return model


def measure_latency(model, sample, runs=200):
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


def save_size_kb(state_dict, tmp_name):
    tmp_path = os.path.join(OUT_DIR, tmp_name)
    torch.save(state_dict, tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)
    return size_kb


def top_k_indices(conv, k):
    """Indices of the k output channels with the largest L1 weight norm, kept in ascending order."""
    scores = conv.weight.detach().abs().sum(dim=(1, 2))
    return sorted(scores.argsort(descending=True)[:k].tolist())


def shrink(model, c1_channels, c2_channels, c3_channels):
    """Build a smaller Conv1DClassifier, keeping each layer's most important filters
    (by L1 norm) and slicing the corresponding BatchNorm params and the next layer's
    input channels to match."""
    c1_keep = top_k_indices(model.conv1, c1_channels)
    c2_keep = top_k_indices(model.conv2, c2_channels)
    c3_keep = top_k_indices(model.conv3, c3_channels)

    new_model = Conv1DClassifier((c1_channels, c2_channels, c3_channels), kernel_sizes, num_classes=len(class_order))

    new_model.conv1.weight.data = model.conv1.weight.data[c1_keep].clone()
    new_model.conv1.bias.data = model.conv1.bias.data[c1_keep].clone()
    new_model.bn1.weight.data = model.bn1.weight.data[c1_keep].clone()
    new_model.bn1.bias.data = model.bn1.bias.data[c1_keep].clone()
    new_model.bn1.running_mean = model.bn1.running_mean[c1_keep].clone()
    new_model.bn1.running_var = model.bn1.running_var[c1_keep].clone()

    new_model.conv2.weight.data = model.conv2.weight.data[c2_keep][:, c1_keep].clone()
    new_model.conv2.bias.data = model.conv2.bias.data[c2_keep].clone()
    new_model.bn2.weight.data = model.bn2.weight.data[c2_keep].clone()
    new_model.bn2.bias.data = model.bn2.bias.data[c2_keep].clone()
    new_model.bn2.running_mean = model.bn2.running_mean[c2_keep].clone()
    new_model.bn2.running_var = model.bn2.running_var[c2_keep].clone()

    new_model.conv3.weight.data = model.conv3.weight.data[c3_keep][:, c2_keep].clone()
    new_model.conv3.bias.data = model.conv3.bias.data[c3_keep].clone()
    new_model.bn3.weight.data = model.bn3.weight.data[c3_keep].clone()
    new_model.bn3.bias.data = model.bn3.bias.data[c3_keep].clone()
    new_model.bn3.running_mean = model.bn3.running_mean[c3_keep].clone()
    new_model.bn3.running_var = model.bn3.running_var[c3_keep].clone()

    new_model.fc.weight.data = model.fc.weight.data[:, c3_keep].clone()
    new_model.fc.bias.data = model.fc.bias.data.clone()

    return new_model


def quantize(model):
    """Static PTQ: fuse Conv+BN+ReLU, calibrate on train data, convert to int8."""
    quant_model = copy.deepcopy(model).eval()
    tq.fuse_modules(
        quant_model,
        [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'], ['conv3', 'bn3', 'relu3']],
        inplace=True,
    )
    torch.backends.quantized.engine = 'qnnpack'
    wrapped = tq.QuantWrapper(quant_model)
    wrapped.qconfig = tq.get_default_qconfig('qnnpack')
    tq.prepare(wrapped, inplace=True)
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(train_loader):
            if i >= CALIBRATION_BATCHES:
                break
            wrapped(X_batch)
    tq.convert(wrapped, inplace=True)
    return wrapped


sample = next(iter(test_loader))[0][:1]
results = []


def nonzero_params(model):
    return sum(int((p != 0).sum().item()) for p in model.parameters())


# ---- Row 0: baseline fp32 (16/32/64, untouched) ----
baseline_acc = accuracy(baseline_model, test_loader)
baseline_params = sum(p.numel() for p in baseline_model.parameters())
baseline_fp32_kb = save_size_kb(baseline_model.state_dict(), '_tmp_baseline_fp32.pt')
baseline_fp32_latency = measure_latency(baseline_model, sample)
# "theoretical flash" = nonzero_params * bytes_per_param, the raw-weight-bytes estimate
# used for comparison (as opposed to the actual torch.save() file, which carries
# fixed pickle/metadata overhead that dominates once models get this small).
baseline_flash_kb = nonzero_params(baseline_model) * 4 / 1024
results.append(('baseline (16/32/64) fp32', baseline_acc, baseline_params, baseline_fp32_kb, baseline_flash_kb, baseline_fp32_latency))
print(f'baseline (16/32/64) fp32: acc={baseline_acc:.4f} params={baseline_params} '
      f'file={baseline_fp32_kb:.2f}KB theoretical_flash={baseline_flash_kb:.2f}KB latency={baseline_fp32_latency:.3f}ms')

# ---- Row 1: baseline + int8 (no structural pruning, quantization only, for reference) ----
baseline_int8 = quantize(baseline_model)
baseline_int8_acc = accuracy(baseline_int8, test_loader)
baseline_int8_kb = save_size_kb(baseline_int8.state_dict(), '_tmp_baseline_int8.pt')
baseline_int8_latency = measure_latency(baseline_int8, sample)
baseline_int8_flash_kb = baseline_params * 1 / 1024  # int8 = 1 byte/param
results.append(('baseline (16/32/64) int8', baseline_int8_acc, baseline_params, baseline_int8_kb, baseline_int8_flash_kb, baseline_int8_latency))
print(f'baseline (16/32/64) int8: acc={baseline_int8_acc:.4f} '
      f'file={baseline_int8_kb:.2f}KB theoretical_flash={baseline_int8_flash_kb:.2f}KB latency={baseline_int8_latency:.3f}ms')

# ---- Shrink + fine-tune + quantize each config ----
smallest_model = None
smallest_name = None

for c1, c2, c3 in SHRINK_CONFIGS:
    name = f'{c1}/{c2}/{c3}'
    shrunk = shrink(baseline_model, c1, c2, c3)

    pre_ft_acc = accuracy(shrunk, test_loader)
    fine_tune(shrunk, FINE_TUNE_EPOCHS, FINE_TUNE_LR)
    post_ft_acc = accuracy(shrunk, test_loader)

    params = sum(p.numel() for p in shrunk.parameters())
    fp32_kb = save_size_kb(shrunk.state_dict(), f'_tmp_shrunk_{c1}_{c2}_{c3}_fp32.pt')
    fp32_flash_kb = params * 4 / 1024
    fp32_latency = measure_latency(shrunk, sample)
    results.append((f'structured {name} fp32', post_ft_acc, params, fp32_kb, fp32_flash_kb, fp32_latency))
    print(f'structured {name} fp32: acc_before_finetune={pre_ft_acc:.4f} acc_after_finetune={post_ft_acc:.4f} '
          f'params={params} file={fp32_kb:.2f}KB theoretical_flash={fp32_flash_kb:.2f}KB latency={fp32_latency:.3f}ms')

    quantized = quantize(shrunk)
    int8_acc = accuracy(quantized, test_loader)
    int8_kb = save_size_kb(quantized.state_dict(), f'_tmp_shrunk_{c1}_{c2}_{c3}_int8.pt')
    int8_flash_kb = params * 1 / 1024
    int8_latency = measure_latency(quantized, sample)
    results.append((f'structured {name} int8', int8_acc, params, int8_kb, int8_flash_kb, int8_latency))
    print(f'structured {name} int8: acc={int8_acc:.4f} file={int8_kb:.2f}KB theoretical_flash={int8_flash_kb:.2f}KB latency={int8_latency:.3f}ms')

    # SHRINK_CONFIGS is ordered largest-to-smallest, so after the loop this holds the smallest.
    smallest_model = quantized
    smallest_name = name

# ---- Summary table ----
print('\n=== Summary ===')
print(f'{"Variant":28s}{"Accuracy":>10s}{"Params":>8s}{"File (KB)":>11s}{"Flash (KB)":>12s}{"Latency (ms)":>14s}')
for name, acc, params, size_kb, flash_kb, latency in results:
    print(f'{name:28s}{acc:>10.4f}{params:>8d}{size_kb:>11.2f}{flash_kb:>12.2f}{latency:>14.3f}')

# ---- Plot: flash size (log scale) and accuracy, grouped by variant family, colored by precision ----
# Two stacked panels sharing one x-axis (not a dual-axis chart) since size (KB,
# log) and accuracy (%) are different scales.
COLOR_FP32 = '#2a78d6'   # categorical slot 1 (blue)
COLOR_INT8 = '#eb6834'   # categorical slot 2 (orange)
GRID_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'
TEXT_SECONDARY = '#52514e'
TEXT_PRIMARY = '#0b0b0b'

families = ['baseline\n(16/32/64)', 'structured\n8/16/32', 'structured\n4/8/16', 'structured\n2/4/8']
by_family = {f: {} for f in families}
for name, acc, params, size_kb, flash_kb, latency in results:
    family = name.replace(' fp32', '').replace(' int8', '')
    family = family.replace('baseline (16/32/64)', 'baseline\n(16/32/64)').replace('structured ', 'structured\n')
    precision = 'int8' if 'int8' in name else 'fp32'
    by_family[family][precision] = (acc, flash_kb)

x = np.arange(len(families))
bar_width = 0.35

fig, (ax_size, ax_acc) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
fig.patch.set_facecolor('#fcfcfb')

fp32_flash = [by_family[f]['fp32'][1] for f in families]
int8_flash = [by_family[f]['int8'][1] for f in families]
fp32_acc = [by_family[f]['fp32'][0] * 100 for f in families]
int8_acc = [by_family[f]['int8'][0] * 100 for f in families]

# -- Top panel: theoretical flash size, log scale --
ax_size.set_facecolor('#fcfcfb')
bars1 = ax_size.bar(x - bar_width / 2, fp32_flash, bar_width, label='fp32', color=COLOR_FP32)
bars2 = ax_size.bar(x + bar_width / 2, int8_flash, bar_width, label='int8', color=COLOR_INT8)
ax_size.set_yscale('log')
ax_size.set_ylabel('Theoretical flash size (KB)\n= nonzero params x bytes/param', color=TEXT_SECONDARY)
ax_size.set_title('Model Size and Accuracy: Structured Pruning + Quantization', color=TEXT_PRIMARY, fontsize=13)
ax_size.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
ax_size.set_axisbelow(True)
for spine in ['top', 'right', 'left']:
    ax_size.spines[spine].set_visible(False)
ax_size.spines['bottom'].set_color(AXIS_COLOR)
ax_size.tick_params(colors=TEXT_SECONDARY)
for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax_size.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2, height),
                          textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8, color=TEXT_PRIMARY)
ax_size.legend(frameon=False, loc='upper right', labelcolor=TEXT_PRIMARY)

# -- Bottom panel: test accuracy --
ax_acc.set_facecolor('#fcfcfb')
bars3 = ax_acc.bar(x - bar_width / 2, fp32_acc, bar_width, label='fp32', color=COLOR_FP32)
bars4 = ax_acc.bar(x + bar_width / 2, int8_acc, bar_width, label='int8', color=COLOR_INT8)
ax_acc.set_ylabel('Test accuracy (%)', color=TEXT_SECONDARY)
ax_acc.set_ylim(0, 108)
ax_acc.set_xticks(x)
ax_acc.set_xticklabels(families, color=TEXT_PRIMARY)
ax_acc.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
ax_acc.set_axisbelow(True)
for spine in ['top', 'right', 'left']:
    ax_acc.spines[spine].set_visible(False)
ax_acc.spines['bottom'].set_color(AXIS_COLOR)
ax_acc.tick_params(colors=TEXT_SECONDARY)
for bars in (bars3, bars4):
    for bar in bars:
        height = bar.get_height()
        ax_acc.annotate(f'{height:.1f}%', (bar.get_x() + bar.get_width() / 2, height),
                         textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8, color=TEXT_PRIMARY)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '17_structured_prune_size_comparison.png'), dpi=150, facecolor='#fcfcfb')
plt.close(fig)

# ---- Confusion matrix for the smallest (most aggressively shrunk) config ----
y_true, y_pred = evaluate(smallest_model, test_loader)
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    [class_order[i] for i in y_true],
    [class_order[i] for i in y_pred],
    labels=class_order, ax=ax,
)
ax.set_title(f'Conv1D Structured-Pruned {smallest_name} + INT8 Confusion Matrix (Test)')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '18_structured_prune_confusion_matrix.png'), dpi=150)
plt.close(fig)

print(classification_report(
    [class_order[i] for i in y_true], [class_order[i] for i in y_pred],
    labels=class_order, zero_division=0,
))

# ---- Save the smallest config's int8 checkpoint ----
torch.save(smallest_model.state_dict(), os.path.join(OUT_DIR, 'conv1d_structured_pruned_int8.pt'))

print('\nSaved size comparison plot, confusion matrix, and conv1d_structured_pruned_int8.pt to', OUT_DIR)
