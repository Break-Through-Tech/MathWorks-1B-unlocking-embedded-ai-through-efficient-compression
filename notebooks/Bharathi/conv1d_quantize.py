import os
import time
import copy

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
PRUNED_CHECKPOINT_PATH = os.path.join(OUT_DIR, 'conv1d_pruned.pt')
INT8_CHECKPOINT_PATH = os.path.join(OUT_DIR, 'conv1d_pruned_int8.pt')

BATCH_SIZE = 32
CALIBRATION_BATCHES = 20


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


# ---- Load the pruned fp32 checkpoint (50% global sparsity, baked in) ----
checkpoint = torch.load(PRUNED_CHECKPOINT_PATH, map_location='cpu')
class_order = checkpoint['class_order']
label_to_idx = {c: i for i, c in enumerate(class_order)}
train_mean = checkpoint['train_mean']
train_std = checkpoint['train_std']

model = Conv1DClassifier(checkpoint['conv_channels'], checkpoint['kernel_sizes'], num_classes=len(class_order))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# ---- Load train (for calibration) and test data, standardized with train stats from the checkpoint ----
train = scipy.io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_train = (train['trainData'] - train_mean) / train_std
y_train = [label[0] for label in train['trainLabels'].flatten()]
X_test = (test['testData'] - train_mean) / train_std
y_test = [label[0] for label in test['testLabels'].flatten()]


def to_dataset(X, y):
    X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    y_t = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)
    return TensorDataset(X_t, y_t)


train_loader = DataLoader(to_dataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(to_dataset(X_test, y_test), batch_size=BATCH_SIZE)


def evaluate(eval_model, loader):
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = eval_model(X_batch).argmax(dim=1).numpy()
            preds_all.extend(preds)
            labels_all.extend(y_batch.numpy())
    return np.array(labels_all), np.array(preds_all)


def measure_latency(eval_model, sample, runs=200):
    with torch.no_grad():
        for _ in range(20):  # warmup
            eval_model(sample)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            eval_model(sample)
            times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


sample = next(iter(test_loader))[0][:1]

# ---- Baseline: pruned fp32, before quantization ----
y_true, y_pred = evaluate(model, test_loader)
fp32_acc = accuracy_score(y_true, y_pred)
fp32_params = sum(p.numel() for p in model.parameters())
fp32_latency = measure_latency(model, sample)

tmp_fp32_path = os.path.join(OUT_DIR, '_tmp_pruned_fp32.pt')
torch.save(model.state_dict(), tmp_fp32_path)
fp32_size_kb = os.path.getsize(tmp_fp32_path) / 1024
os.remove(tmp_fp32_path)

print('=== Pruned fp32 (before quantization) ===')
print(f'Accuracy: {fp32_acc:.4f}')
print(f'Params: {fp32_params}')
print(f'Size on disk: {fp32_size_kb:.2f} KB')
print(f'Latency (median, ms): {fp32_latency:.3f}\n')

# ---- Fuse Conv+BN+ReLU per block (no BN/ReLU after fc, nothing to fuse there) ----
quant_model = copy.deepcopy(model)
quant_model.eval()
tq.fuse_modules(
    quant_model,
    [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'], ['conv3', 'bn3', 'relu3']],
    inplace=True,
)

# ---- Static post-training quantization: wrap, configure, calibrate, convert ----
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

# ---- Evaluate the pruned + int8 model ----
y_true, y_pred = evaluate(wrapped, test_loader)
int8_acc = accuracy_score(y_true, y_pred)
int8_latency = measure_latency(wrapped, sample)

torch.save(wrapped.state_dict(), INT8_CHECKPOINT_PATH)
int8_size_kb = os.path.getsize(INT8_CHECKPOINT_PATH) / 1024

print('=== Pruned + int8 quantized ===')
print(f'Accuracy: {int8_acc:.4f}')
print(f'Size on disk: {int8_size_kb:.2f} KB')
print(f'Latency (median, ms): {int8_latency:.3f}\n')

print(classification_report(
    [class_order[i] for i in y_true], [class_order[i] for i in y_pred],
    labels=class_order, zero_division=0,
))

# ---- Confusion matrix ----
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    [class_order[i] for i in y_true],
    [class_order[i] for i in y_pred],
    labels=class_order, ax=ax,
)
ax.set_title('Conv1D Pruned + INT8 Quantized Confusion Matrix (Test)')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '16_pruned_quantized_confusion_matrix.png'), dpi=150)
plt.close(fig)

print('=== Summary: pruned fp32 vs. pruned + int8 ===')
print(f'{"":20s}{"fp32 (pruned)":>15s}{"int8 (pruned)":>15s}')
print(f'{"Accuracy":20s}{fp32_acc:>15.4f}{int8_acc:>15.4f}')
print(f'{"Size (KB)":20s}{fp32_size_kb:>15.2f}{int8_size_kb:>15.2f}')
print(f'{"Latency (ms)":20s}{fp32_latency:>15.3f}{int8_latency:>15.3f}')

print('\nSaved confusion matrix and conv1d_pruned_int8.pt to', OUT_DIR)
