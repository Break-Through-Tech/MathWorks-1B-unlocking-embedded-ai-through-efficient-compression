import torch
import torch.nn as nn
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

HERE = Path(__file__).resolve().parent

# Settings
DATA_FILE = "standardized_global.npz"
EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 32
CLASS_NAMES = ["Normal", "InnerRaceFault", "OuterRaceFault"]

torch.manual_seed(2)

# Load data
data = np.load(HERE / DATA_FILE)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

def make_loader(X, y, shuffle):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # (N, 5000) -> (N, 1, 5000)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)

# Building the neural network
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Look for patterns in the signal
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 16, kernel_size=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Turn those patterns into one score per class
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def predict(loader):
    """Return the model's guesses and the true labels for a data loader"""
    model.eval()
    guesses, truth = [], []
    with torch.no_grad():
        for X, y in loader:
            guesses.append(model(X).argmax(dim=1))
            truth.append(y)
    return torch.cat(guesses), torch.cat(truth)

def get_accuracy(loader):
    """Fraction of signals in this loader that the model gets right"""
    guesses, truth = predict(loader)
    return (guesses == truth).float().mean().item()

# Train model
for epoch in range(1, EPOCHS + 1):
    model.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}: val accuracy {get_accuracy(val_loader):.3f}")

# Results
test_accuracy = get_accuracy(test_loader)
print(f"test accuracy: {test_accuracy:.3f}")

guesses, truth = predict(test_loader)
matrix = confusion_matrix(truth, guesses)
print(matrix)
ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(xticks_rotation=20)
plt.tight_layout()
plt.savefig(HERE / "confusion_matrix.png", dpi=150)

# Baseline numbers for compression
num_params = sum(p.numel() for p in model.parameters())

model_path = HERE / "cnn_baseline.pt"
torch.save(model.state_dict(), model_path)
size_kb = model_path.stat().st_size / 1024

# Time one signal at a time, after a short warm-up
sample = next(iter(test_loader))[0][:1]  # shape (1, 1, 5000)
model.eval()
with torch.no_grad():
    for _ in range(20):
        model(sample)
    start = time.perf_counter()
    for _ in range(200):
        model(sample)
ms_per_signal = (time.perf_counter() - start) / 200 * 1000

print("\nBASELINE")
print(f"  Test accuracy: {test_accuracy:.3f}")
print(f"  Parameters:    {num_params:,}")
print(f"  File size:     {size_kb:.1f} KB")
print(f"  Speed:         {ms_per_signal:.3f} ms per signal")
print(f"  Training:      {EPOCHS} epochs, lr {LEARNING_RATE}, batch size {BATCH_SIZE}")