import os

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR
CHECKPOINT_PATH = os.path.join(OUT_DIR, 'conv1d_fp32.pt')

BATCH_SIZE = 32
SPARSITY_LEVELS = [i / 100 for i in range(5, 100, 5)]  # 5%, 10%, ..., 95%


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


# ---- Load checkpoint + test data once ----
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
class_order = checkpoint['class_order']
label_to_idx = {c: i for i, c in enumerate(class_order)}
train_mean = checkpoint['train_mean']
train_std = checkpoint['train_std']

test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))
X_test = (test['testData'] - train_mean) / train_std
y_test = [label[0] for label in test['testLabels'].flatten()]


def to_dataset(X, y):
    X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    y_t = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)
    return TensorDataset(X_t, y_t)


test_loader = DataLoader(to_dataset(X_test, y_test), batch_size=BATCH_SIZE)


def build_model():
    """Fresh, unpruned copy of the trained model, reloaded from the checkpoint."""
    model = Conv1DClassifier(checkpoint['conv_channels'], checkpoint['kernel_sizes'], num_classes=len(class_order))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def test_accuracy(model):
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).argmax(dim=1).numpy()
            preds_all.extend(preds)
            labels_all.extend(y_batch.numpy())
    return accuracy_score(labels_all, preds_all)


# ---- Baseline (0% pruned) ----
baseline_acc = test_accuracy(build_model())
print(f'  0% pruned: test accuracy = {baseline_acc:.4f}')

# ---- Sweep: fresh model + fresh global pruning at each sparsity level ----
# Each level starts from the original fp32 weights (not the previous level's
# pruned model), so this measures one-shot pruning at each sparsity in isolation.
sparsities, accuracies = [0.0], [baseline_acc]
for sparsity in SPARSITY_LEVELS:
    model = build_model()
    prunable = [(model.conv1, 'weight'), (model.conv2, 'weight'), (model.conv3, 'weight'), (model.fc, 'weight')]
    prune.global_unstructured(prunable, pruning_method=prune.L1Unstructured, amount=sparsity)

    acc = test_accuracy(model)
    sparsities.append(sparsity)
    accuracies.append(acc)
    print(f'{int(sparsity * 100):3d}% pruned: test accuracy = {acc:.4f}')

# ---- Plot: accuracy vs. sparsity ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([s * 100 for s in sparsities], accuracies, marker='o')
ax.set_xlabel('Global sparsity (%)')
ax.set_ylabel('Test accuracy')
ax.set_title('Conv1D Test Accuracy vs. Pruning Sparsity (one-shot, global L1)')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '15_pruning_sparsity_sweep.png'), dpi=150)
plt.close(fig)

print('\nSaved sparsity sweep plot to', OUT_DIR)
