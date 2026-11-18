import os

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR

class_order = ['InnerRaceFault', 'Normal', 'OuterRaceFault']
label_to_idx = {c: i for i, c in enumerate(class_order)}

# ---- Hyperparameters (tune these) ----
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
CONV_CHANNELS = [16, 32, 64]   # output channels for conv layers 1, 2, 3
KERNEL_SIZES = [7, 5, 3]       # kernel size for conv layers 1, 2, 3
WEIGHT_DECAY = 0.0
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print('Using device:', device)

# ---- Load data ----
train = scipy.io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
val = scipy.io.loadmat(os.path.join(DATA_DIR, 'val.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_train = train['trainData']
y_train = [label[0] for label in train['trainLabels'].flatten()]
X_val = val['valData']
y_val = [label[0] for label in val['valLabels'].flatten()]
X_test = test['testData']
y_test = [label[0] for label in test['testLabels'].flatten()]

# ---- Global standardization (train stats only) ----
train_mean = np.mean(X_train)
train_std = np.std(X_train)

X_train_std = (X_train - train_mean) / train_std
X_val_std = (X_val - train_mean) / train_std
X_test_std = (X_test - train_mean) / train_std

# ---- Build datasets: reshape to (N, 1, 5000) for Conv1d (1 input channel) ----
# Cast to float32 in numpy BEFORE torch.tensor() - converting a large float64
# array via torch's own dtype cast segfaults with this torch/numpy combo.
def to_dataset(X, y):
    X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    y_t = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)
    return TensorDataset(X_t, y_t)

train_loader = DataLoader(to_dataset(X_train_std, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(to_dataset(X_val_std, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(to_dataset(X_test_std, y_test), batch_size=BATCH_SIZE)

# ---- Model: simple 3-layer Conv1D classifier ----
class Conv1DClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        c1, c2, c3 = CONV_CHANNELS
        k1, k2, k3 = KERNEL_SIZES

        self.conv1 = nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2)
        # Separate ReLU/pool instances rather than one reused module: eager-mode
        # quantization fuses Conv+ReLU by module pair, and one shared instance
        # cannot be fused into three different convs.
        self.relu1, self.relu2, self.relu3 = nn.ReLU(), nn.ReLU(), nn.ReLU()
        self.pool1, self.pool2 = nn.MaxPool1d(2), nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

model = Conv1DClassifier(num_classes=len(class_order)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# ---- Helper: average loss/accuracy over a loader ----
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss_sum += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)
    return loss_sum / total, correct / total

# ---- Training loop ----
# Only validation is tracked per epoch - test set stays untouched until the end.
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss, running_total = 0.0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        running_total += X_batch.size(0)

    train_loss = running_loss / running_total
    val_loss, val_acc = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f'Epoch {epoch:3d}/{NUM_EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

# ---- Training curves ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses, label='Validation')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(val_accuracies)
axes[1].set_title('Validation Accuracy')
axes[1].set_xlabel('Epoch')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '10_conv1d_training_curves.png'), dpi=150)
plt.close(fig)

# ---- Final evaluation: validation again, and test ONCE ----
def predict_all(loader):
    preds_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(y_batch.numpy())
    return np.array(labels_all), np.array(preds_all)

y_val_true, y_val_pred = predict_all(val_loader)
print('\n--- Validation ---')
print(f'Accuracy: {accuracy_score(y_val_true, y_val_pred):.4f}')
print(classification_report(y_val_true, y_val_pred, target_names=class_order, zero_division=0))

y_test_true, y_test_pred = predict_all(test_loader)
print('--- Test ---')
print(f'Accuracy: {accuracy_score(y_test_true, y_test_pred):.4f}')
print(classification_report(y_test_true, y_test_pred, target_names=class_order, zero_division=0))

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    [class_order[i] for i in y_test_true],
    [class_order[i] for i in y_test_pred],
    labels=class_order, ax=ax,
)
ax.set_title('Conv1D Confusion Matrix (Test)')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '11_conv1d_confusion_matrix.png'), dpi=150)
plt.close(fig)

# ---- Save FP32 checkpoint ----
# Normalization stats travel with the weights so downstream scripts (e.g.
# quantization) reproduce the exact same preprocessing from train stats alone.
torch.save({
    'state_dict': model.state_dict(),
    'conv_channels': CONV_CHANNELS,
    'kernel_sizes': KERNEL_SIZES,
    'class_order': class_order,
    'train_mean': float(train_mean),
    'train_std': float(train_std),
}, os.path.join(OUT_DIR, 'conv1d_fp32.pt'))

print('Saved training curves, confusion matrix and conv1d_fp32.pt to', OUT_DIR)
