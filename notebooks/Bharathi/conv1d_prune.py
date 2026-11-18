import os

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR
CHECKPOINT_PATH = os.path.join(OUT_DIR, 'conv1d_fp32.pt')
PRUNED_CHECKPOINT_PATH = os.path.join(OUT_DIR, 'conv1d_pruned.pt')

# ---- Hyperparameters ----
BATCH_SIZE = 32
GLOBAL_SPARSITY = 0.5  # fraction of weights to zero out, globally across conv1/conv2/conv3/fc


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


# ---- Load checkpoint ----
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
class_order = checkpoint['class_order']
label_to_idx = {c: i for i, c in enumerate(class_order)}
train_mean = checkpoint['train_mean']
train_std = checkpoint['train_std']

model = Conv1DClassifier(checkpoint['conv_channels'], checkpoint['kernel_sizes'], num_classes=len(class_order))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# ---- Load val/test data, standardize with the same train stats the checkpoint was trained on ----
val = scipy.io.loadmat(os.path.join(DATA_DIR, 'val.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_val = (val['valData'] - train_mean) / train_std
y_val = [label[0] for label in val['valLabels'].flatten()]
X_test = (test['testData'] - train_mean) / train_std
y_test = [label[0] for label in test['testLabels'].flatten()]


def to_dataset(X, y):
    X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    y_t = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)
    return TensorDataset(X_t, y_t)


val_loader = DataLoader(to_dataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(to_dataset(X_test, y_test), batch_size=BATCH_SIZE)


def predict_all(loader):
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch).argmax(dim=1).numpy()
            preds_all.extend(preds)
            labels_all.extend(y_batch.numpy())
    return np.array(labels_all), np.array(preds_all)


def evaluate(name):
    y_val_true, y_val_pred = predict_all(val_loader)
    y_test_true, y_test_pred = predict_all(test_loader)

    print(f'--- {name}: Validation ---')
    print(f'Accuracy: {accuracy_score(y_val_true, y_val_pred):.4f}')
    print(classification_report(y_val_true, y_val_pred, target_names=class_order, zero_division=0))

    print(f'--- {name}: Test ---')
    print(f'Accuracy: {accuracy_score(y_test_true, y_test_pred):.4f}')
    print(classification_report(y_test_true, y_test_pred, target_names=class_order, zero_division=0))

    return y_test_true, y_test_pred


# ---- Baseline (before pruning) ----
print('=== Baseline (fp32, unpruned) ===\n')
evaluate('Baseline')

# ---- Global unstructured L1 pruning across conv1/conv2/conv3/fc ----
# Named tuples so later sparsity reporting and prune.remove() can reuse the same list.
prunable = [
    ('conv1', model.conv1),
    ('conv2', model.conv2),
    ('conv3', model.conv3),
    ('fc', model.fc),
]

prune.global_unstructured(
    [(module, 'weight') for _, module in prunable],
    pruning_method=prune.L1Unstructured,
    amount=GLOBAL_SPARSITY,
)

# ---- Per-layer sparsity after pruning ----
print(f'=== Per-layer sparsity after global {int(GLOBAL_SPARSITY * 100)}% L1 pruning ===')
total_params, total_zeros = 0, 0
for name, module in prunable:
    weight = module.weight
    zeros = int(torch.sum(weight == 0))
    n = weight.nelement()
    total_params += n
    total_zeros += zeros
    print(f'{name}: {zeros}/{n} zeros ({100 * zeros / n:.1f}%)')
print(f'Overall (pruned layers only): {total_zeros}/{total_params} zeros ({100 * total_zeros / total_params:.1f}%)\n')

# ---- Evaluate after pruning (masks already applied via the weight_orig/weight_mask hooks) ----
print(f'=== Pruned ({int(GLOBAL_SPARSITY * 100)}% global sparsity) ===\n')
y_test_true, y_test_pred = evaluate('Pruned')

# ---- Make pruning permanent: drop weight_orig/weight_mask, keep zeroed weight as a plain tensor ----
for _, module in prunable:
    prune.remove(module, 'weight')

# ---- Confusion matrix (pruned model, test set) ----
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    [class_order[i] for i in y_test_true],
    [class_order[i] for i in y_test_pred],
    labels=class_order, ax=ax,
)
ax.set_title(f'Conv1D Pruned ({int(GLOBAL_SPARSITY * 100)}% sparsity) Confusion Matrix (Test)')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '14_pruned_confusion_matrix.png'), dpi=150)
plt.close(fig)

# ---- Save pruned checkpoint ----
torch.save({
    'state_dict': model.state_dict(),
    'conv_channels': checkpoint['conv_channels'],
    'kernel_sizes': checkpoint['kernel_sizes'],
    'class_order': class_order,
    'train_mean': train_mean,
    'train_std': train_std,
    'global_sparsity': GLOBAL_SPARSITY,
}, PRUNED_CHECKPOINT_PATH)

print('Saved pruned confusion matrix and conv1d_pruned.pt to', OUT_DIR)
