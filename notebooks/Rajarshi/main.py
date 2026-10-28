import os
# The mathworks1b env ships two OpenMP libraries (conda's and torch's); multi-threaded
# torch segfaults when both load, so pin to one thread. Must be set before torch is imported.
os.environ['OMP_NUM_THREADS'] = '1'

import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from scipy import io
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Build paths from this file's location so the script runs from any working directory
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'
OUTPUT_DIR = Path(__file__).resolve().parent     # images are saved next to this script

train = DATA_DIR / 'train.mat'
test = DATA_DIR / 'test.mat'
val = DATA_DIR / 'val.mat'

train_data = io.loadmat(train)
test_data = io.loadmat(test)
val_data = io.loadmat(val)

# Filter out metadata
train_clean_data = {k: v for k, v in train_data.items() if not (k.startswith('__') and k.endswith('__'))}
# Filter out metadata
test_clean_data = {k: v for k, v in test_data.items() if not (k.startswith('__') and k.endswith('__'))}
val_clean_data = {k: v for k, v in val_data.items() if not (k.startswith('__') and k.endswith('__'))}

# Confirm the filter only dropped the metadata keys, not any actual values
def check_clean_matches_original(clean_data, original_data, labels_key):
    for key, value in clean_data.items():
        if key == labels_key:
            same = all(a[0] == b[0] for a, b in zip(value, original_data[key]))
        else:
            same = np.array_equal(value, original_data[key])
        print(f'{key}: {"match" if same else "MISMATCH"}')
# check_clean_matches_original(train_clean_data, train_data, 'trainLabels')

X_train = train_data['trainData']       # shape: (393, 5000)
X_test = test_data['testData']
X_val = val_data['valData']

y_train_npstr = [label[0] for label in train_data['trainLabels'].flatten()]
y_test_npstr = [label[0] for label in test_data['testLabels'].flatten()]
y_val_npstr = [label[0] for label in val_data['valLabels'].flatten()]

y_train = [i.astype(str).tolist() for i in y_train_npstr]
y_test = [i.astype(str).tolist() for i in y_test_npstr]
y_val = [i.astype(str).tolist() for i in y_val_npstr]

# ----------------------------------
# 
# Standardizing the Data
#
#----------------------------------

# We start with samlpe-wise z-score standardization
# Each sample will also have metadata, which would contain certain calculations that determine the properties of that sample
# The metadata can then be used as 'features' for the ML model
def standardize_samplewise(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / std

# Standardize the training set, validation set and test set
# since this is sample wise there is no data leakage
# We do this so that the model tries to find patterns in the actual features and differences of the sapmle and not the noise
X_train_std = standardize_samplewise(X_train)
X_val_std = standardize_samplewise(X_val)
X_test_std  = standardize_samplewise(X_test)

# Calculating metadata for each sample
# Shape of metadata would be an array of 7 values per sample
# This metadata acts as features for the sample and input features for the model
# [Mean, RMS, Peak Amplitude, Crest Factor, Standard Deviation, Kurtosis, Skewness]
X_train_metadata = np.zeros((X_train.shape[0], 7))
X_test_metadata = np.zeros((X_test.shape[0], 7))
X_val_metadata = np.zeros((X_val.shape[0], 7))

def metadata_extraction(dataset, meta_dataset):
    for idx, i in enumerate(dataset):
        mean = np.mean(i)
        RMS = np.sqrt(np.mean(i**2))
        peak = np.max(np.abs(i))
        crest = peak/RMS
        std = np.std(i)     # np.sqrt(sum((i - mean)**2) / len(i))
        kurtosis = (np.mean((i - mean)**4) / std**4) - 3    # (((1/len(i)) * sum( (i - mean)**4 )) / std**4) - 3
        skewness = np.mean((i - mean)**3) / std**3      # ((1/len(i)) * sum((i - mean)**3)) / std**3

        meta_dataset[idx] = [mean, RMS, peak, crest, std, kurtosis, skewness]

metadata_extraction(X_train, X_train_metadata)
metadata_extraction(X_test, X_test_metadata)
metadata_extraction(X_val, X_val_metadata)

# Standardizing the feature metadata for training, test and validation samples
# All the metadatas have varying scale, so we standardize them to put them on the same scale (mean = 0, std = 1)
# This makes sure no feature with a larger scale dominates the neural networks and all features are treated equally
train_mean = X_train_metadata.mean(axis=0)
train_std = X_train_metadata.std(axis=0)

# We standardize using the training data's mean and std as we don't want data to leak into the test and val
# If we standardized using the test and val data respectively, we would be assuming something about the data
# During inference, when we standardize a single sample, the standardization variables have to come from somewhere
# That somewhere can only be from a place we already know, which is only the training data
X_train_metadata_standard = (X_train_metadata - train_mean) / train_std
X_test_metadata_standard = (X_test_metadata - train_mean) / train_std
X_val_metadata_standard = (X_val_metadata - train_mean) / train_std


#-----------------------------------------
#
# Model Selection
#
#----------------------------------------

# All three models take the sample-wise standardized signals (X_*_std), not the metadata
# The metadata is 7 separate statistics, not a sequence, so there is nothing for a CNN/LSTM/GRU to scan over

classes = sorted(set(y_train))      # ['InnerRaceFault', 'Normal', 'OuterRaceFault']
class_to_idx = {c: i for i, c in enumerate(classes)}

# X_*_std is Fortran-ordered float64 (inherited from the .mat file); torch.tensor on it segfaults
# in this env (torch 2.3.1 + numpy 2.5), so convert to C-contiguous float32 first
def to_tensors(X, y):
    X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    y = torch.tensor([class_to_idx[label] for label in y], dtype=torch.long)
    return X, y

X_train_t, y_train_t = to_tensors(X_train_std, y_train)
X_val_t, y_val_t = to_tensors(X_val_std, y_val)
X_test_t, y_test_t = to_tensors(X_test_std, y_test)

# Input shapes
# CNN:      (N, 1, 5000)  -> 1 channel, the whole signal; filters slide along all 5000 readings
# LSTM/GRU: (N, 50, 100)  -> 50 consecutive frames of 100 readings; the model steps through the frames in order
# Same 5000 readings in the same order either way, just grouped differently
N_FRAMES, FRAME_LEN = 50, 100

def cnn_input(X):
    return X.unsqueeze(1)

def rnn_input(X):
    return X.reshape(X.shape[0], N_FRAMES, FRAME_LEN)


class FaultCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x

    def loss_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        return criterion, optimizer


# LSTM: reads the 50 frames one at a time, updating its hidden state (memory) after each
# The hidden state after the last frame summarizes the whole signal and is used to classify
class FaultLSTM(nn.Module):
    def __init__(self, input_size=FRAME_LEN, hidden_size=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.rnn(x)        # out: (batch, 50, hidden_size)
        x = self.fc(out[:, -1])     # classify from the last time step

        return x

    def loss_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        return criterion, optimizer


# GRU: same idea as the LSTM with a simpler cell (2 gates instead of 3, so fewer parameters)
class FaultGRU(nn.Module):
    def __init__(self, input_size=FRAME_LEN, hidden_size=32):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.rnn(x)
        x = self.fc(out[:, -1])

        return x

    def loss_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        return criterion, optimizer


@torch.no_grad()
def evaluate(model, X, y, criterion):
    model.eval()
    logits = model(X)
    loss = criterion(logits, y).item()
    pred = logits.argmax(dim=1)
    return loss, (pred == y).float().mean().item(), pred

# Shared training loop so all three models are trained the same way
# Keeps the weights from the epoch with the lowest validation loss
def train_model(model, X_tr, y_tr, X_va, y_va, epochs=150, batch_size=32, seed=42, verbose=True):
    criterion, optimizer = model.loss_optimizer()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_loss, best_state, best_epoch = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        tr_loss, tr_acc, _ = evaluate(model, X_tr, y_tr, criterion)
        va_loss, va_acc, _ = evaluate(model, X_va, y_va, criterion)
        if va_loss < best_loss:
            best_loss, best_state, best_epoch = va_loss, copy.deepcopy(model.state_dict()), epoch
        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(f"  epoch {epoch:3d}  train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")

    model.load_state_dict(best_state)
    return best_epoch

def report(name, model, best_epoch, X_tr, X_va, X_te):
    criterion = nn.CrossEntropyLoss()
    _, tr_acc, _ = evaluate(model, X_tr, y_train_t, criterion)
    _, va_acc, _ = evaluate(model, X_va, y_val_t, criterion)
    _, te_acc, te_pred = evaluate(model, X_te, y_test_t, criterion)
    pred_labels = [classes[i] for i in te_pred.tolist()]

    print(f"\n{name} - Test Set")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}, best epoch: {best_epoch}")
    print(f"Train accuracy: {tr_acc:.4f}")
    print(f"Val accuracy:   {va_acc:.4f}")
    print(f"Test accuracy:  {te_acc:.4f}\n")
    print(classification_report(y_test, pred_labels))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, pred_labels, labels=classes),
                       index=classes, columns=classes))
    print()


models = [
    ("FaultCNN", FaultCNN, cnn_input),
    ("FaultLSTM", FaultLSTM, rnn_input),
    ("FaultGRU", FaultGRU, rnn_input),
]

trained_models = {}
for name, model_class, shape_input in models:
    torch.manual_seed(42)       # same seed per model so each result is reproducible on its own
    model = model_class()
    X_tr, X_va, X_te = shape_input(X_train_t), shape_input(X_val_t), shape_input(X_test_t)
    print(f"Training {name} on input shape {tuple(X_tr.shape[1:])}")
    best_epoch = train_model(model, X_tr, y_train_t, X_va, y_val_t)
    report(name, model, best_epoch, X_tr, X_va, X_te)
    trained_models[name] = model


#-----------------------------------------
#
# Pruning (CNN only)
#
#----------------------------------------

# Standard (global, unstructured, L1-magnitude) pruning
# At each level, the smallest-magnitude weights across conv1, conv2 and fc are ranked together
# and that percentage is set to zero. Biases are left alone.
# Each level starts from a fresh copy of the trained CNN and is evaluated two ways:
#   One-shot:   prune and evaluate straight away (no retraining)
#   Fine-tuned: prune, then retrain for FINETUNE_EPOCHS so the remaining weights can compensate
# During fine-tuning the pruning mask stays attached, so pruned weights are held at zero
PRUNE_PERCENTAGES = [0, 20, 40, 60, 62, 64, 66, 68, 70, 80, 90, 92, 94, 96, 98, 100]
FINETUNE_EPOCHS = 20

trained_cnn = trained_models["FaultCNN"]
X_tr_cnn, X_va_cnn, X_te_cnn = cnn_input(X_train_t), cnn_input(X_val_t), cnn_input(X_test_t)
criterion = nn.CrossEntropyLoss()

def prunable(model):
    return [(model.conv1, 'weight'), (model.conv2, 'weight'), (model.fc, 'weight')]

def prune_cnn(model, percent):
    pruned = copy.deepcopy(model)
    prune.global_unstructured(prunable(pruned), pruning_method=prune.L1Unstructured, amount=percent / 100)
    return pruned       # mask still attached: weight = weight_orig * weight_mask

def make_permanent(model):
    for module, param in prunable(model):
        prune.remove(module, param)     # bake the zeros into the weights and drop the mask
    return model

def count_weights_left(model):
    return sum(int((module.weight != 0).sum()) for module, _ in prunable(model))

def accuracies(model):
    return [evaluate(model, X, y, criterion)[1]
            for X, y in ((X_tr_cnn, y_train_t), (X_va_cnn, y_val_t), (X_te_cnn, y_test_t))]

total_weights = sum(module.weight.numel() for module, _ in prunable(trained_cnn))
pruning_results = []

print(f"FaultCNN Pruning ({total_weights} prunable weights, fine-tuning {FINETUNE_EPOCHS} epochs per level)")
print(f"{'':>7} {'':>13} {'One-shot':>8} | {'Fine-tuned':^27}")
print(f"{'Pruned':>7} {'Weights left':>13} {'Test':>8} | {'Train':>7} {'Val':>7} {'Test':>7} {'Left':>5}")
for percent in PRUNE_PERCENTAGES:
    # One-shot
    oneshot_cnn = make_permanent(prune_cnn(trained_cnn, percent))
    weights_left = count_weights_left(oneshot_cnn)
    _, _, oneshot_te = accuracies(oneshot_cnn)

    # Fine-tuned (same pruned starting point, mask kept on while training)
    torch.manual_seed(42)
    finetuned_cnn = prune_cnn(trained_cnn, percent)
    train_model(finetuned_cnn, X_tr_cnn, y_train_t, X_va_cnn, y_val_t, epochs=FINETUNE_EPOCHS, verbose=False)
    finetuned_cnn = make_permanent(finetuned_cnn)
    ft_tr, ft_va, ft_te = accuracies(finetuned_cnn)
    ft_left = count_weights_left(finetuned_cnn)     # should equal weights_left (pruned weights stayed zero)

    pruning_results.append((percent, weights_left, oneshot_te, ft_tr, ft_va, ft_te))
    print(f"{percent:>6}% {weights_left:>13} {oneshot_te:>8.4f} | {ft_tr:>7.4f} {ft_va:>7.4f} {ft_te:>7.4f} {ft_left:>5}")
print()

# Pruning percentage vs test accuracy, one-shot vs fine-tuned
percents = [r[0] for r in pruning_results]
oneshot_accs = [r[2] * 100 for r in pruning_results]
finetuned_accs = [r[5] * 100 for r in pruning_results]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(100 / 3, color='#a3a29b', linewidth=1, linestyle='--')
ax.text(1, 100 / 3 + 1.5, 'Chance (33%)', color='#7a7972', fontsize=9)
ax.plot(percents, oneshot_accs, color='#2a78d6', linewidth=2, marker='o', markersize=6,
        markeredgecolor='#fcfcfb', markeredgewidth=1.5, label='Pruned only (one-shot)')
ax.plot(percents, finetuned_accs, color='#eb6834', linewidth=2, marker='s', markersize=6,
        markeredgecolor='#fcfcfb', markeredgewidth=1.5, label=f'Pruned + fine-tuned ({FINETUNE_EPOCHS} epochs)')
ax.set_title('FaultCNN: Pruning Percentage vs Test Accuracy', color='#0b0b0b', fontsize=13, pad=12)
ax.set_xlabel('Weights pruned (%)', color='#52514e')
ax.set_ylabel('Test accuracy (%)', color='#52514e')
ax.set_xlim(-2, 102)
ax.set_ylim(0, 105)
ax.set_xticks(range(0, 101, 10))
ax.legend(frameon=False, labelcolor='#52514e', loc='lower left')
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
plt.savefig(OUTPUT_DIR / 'cnn_pruning_accuracy.png', dpi=150)

print("Original shape:", X_train.shape)
print("Standardized shape:", X_train_std.shape)
print("Mean of the first sample after scaling:", np.mean(X_train_std[0])) # Should be ~0
print("Std of the first sample after scaling:", np.std(X_train_std[0]))
print()

# Verify the standardization by checking the pooled distribution against a standard normal
flat = X_train_std.ravel()
print("Overall mean:", flat.mean(), "(expect ~0)")
print("Overall std:", flat.std(), "(expect ~1)")
print("Skewness:", stats.skew(flat), "(expect ~0 for a symmetric bell curve)")
print("Excess kurtosis:", stats.kurtosis(flat), "(expect ~0 for a normal-shaped tail)")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(flat, bins=100, density=True, color='#2a78d6', alpha=0.75, edgecolor='none', label='Standardized data')
x = np.linspace(flat.min(), flat.max(), 400)
ax.plot(x, stats.norm.pdf(x, 0, 1), color='#e34948', linewidth=1.5, label='Standard normal N(0,1)')
ax.set_title('Z-score Distribution (X_standardized)', color='#0b0b0b', fontsize=13, pad=12)
ax.set_xlabel('Z-score', color='#52514e')
ax.set_ylabel('Density', color='#52514e')
ax.legend(frameon=False, labelcolor='#52514e')
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
plt.savefig(OUTPUT_DIR / 'standardized_zscore_histogram.png', dpi=150)
plt.show()

