# Trains the spectrogram CNN across 5 seeds and saves results
import torch
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import SpecCNN
from common import make_loader, run_seeds, predict

HERE = Path(__file__).resolve().parent

# Settings
DATA_FILE = "spectrograms_128.npz"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
SEEDS = [0, 1, 2, 3, 4]
CLASS_NAMES = ["Normal", "InnerRaceFault", "OuterRaceFault"]
CONV1_FILTERS = 8
TAG = "baseline"

data = np.load(HERE / "processed_data" / DATA_FILE)
train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)

def make_model():
    return SpecCNN(conv1_filters=CONV1_FILTERS)

test_accs, models = run_seeds(make_model, train_loader, val_loader, test_loader, SEEDS, EPOCHS, LEARNING_RATE)
for seed, m in zip(SEEDS, models):
    torch.save(m.state_dict(), HERE / "weights" / f"spec_cnn_{TAG}_seed{seed}.pt")
model = models[-1]
print("parameters:", sum(p.numel() for p in model.parameters()))
print(f"mean {np.mean(test_accs):.3f}, std {np.std(test_accs):.3f}")
(HERE / "results" / f"results_spec_{TAG}.json").write_text(json.dumps({"test_accs": test_accs}))

guesses, truth = predict(model, test_loader)
matrix = confusion_matrix(truth, guesses)
print(matrix)
ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(xticks_rotation=20)
plt.tight_layout()
plt.savefig(HERE / "figures" / f"confusion_matrix_spec_{TAG}.png", dpi=150)
