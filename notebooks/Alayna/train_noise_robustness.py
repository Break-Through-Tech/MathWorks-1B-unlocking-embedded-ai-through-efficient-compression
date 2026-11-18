from pathlib import Path
import copy
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
CLASSES = ["Normal", "InnerRaceFault", "OuterRaceFault"]
NOISE_LEVELS_DB = [30, 20, 10, 5, 0, -5]
REPEATS = 10

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(torch.get_num_threads(), 4))
DEVICE = torch.device("cpu")
OUTPUT_DIR = Path.cwd() / "robustness_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

candidates = [Path.cwd() / "data", Path.cwd() / "../../data", Path.cwd() / "../data", Path.cwd(), Path("/mnt/data")]
DATA_DIR = next((p.resolve() for p in candidates if all((p / f"{s}.mat").is_file() for s in ("train", "val", "test"))), None)
if DATA_DIR is None:
    raise FileNotFoundError("Place train.mat, val.mat, and test.mat in data/ or beside this script.")


def load_split(split):
    mat = loadmat(DATA_DIR / f"{split}.mat")
    X = np.asarray(mat[f"{split}Data"], dtype=np.float32)
    labels = [str(np.ravel(v)[0]) for v in mat[f"{split}Labels"].ravel()]
    y = np.asarray([CLASSES.index(label) for label in labels], dtype=np.int64)
    assert X.ndim == 2 and X.shape[1] == 5000 and np.isfinite(X).all()
    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")
train_mean = float(X_train.mean())
train_std = float(X_train.std())
assert train_std > 0


def to_tensor(raw_signals):
    standardized = (raw_signals - train_mean) / train_std
    return torch.as_tensor(standardized[:, None, :], dtype=torch.float32)


train_loader = DataLoader(
    TensorDataset(to_tensor(X_train), torch.as_tensor(y_train, dtype=torch.long)),
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
val_X_tensor = to_tensor(X_val)
val_y_tensor = torch.as_tensor(y_val, dtype=torch.long)


class SmallBearingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, len(CLASSES)),
        )

    def forward(self, x):
        return self.layers(x)


model = SmallBearingCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_parameters = sum(p.numel() for p in model.parameters())
best_val_loss = float("inf")
best_epoch = 0
best_weights = None
stale_epochs = 0

for epoch in range(1, 41):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_loss = criterion(model(val_X_tensor.to(DEVICE)), val_y_tensor.to(DEVICE)).item()

    if validation_loss < best_val_loss - 1e-4:
        best_val_loss = validation_loss
        best_epoch = epoch
        best_weights = copy.deepcopy(model.state_dict())
        stale_epochs = 0
    else:
        stale_epochs += 1
    if stale_epochs >= 8:
        break

model.load_state_dict(best_weights)
model.eval()
torch.save(
    {"state_dict": model.state_dict(), "mean": train_mean, "std": train_std,
     "class_names": CLASSES, "parameters": num_parameters, "seed": SEED},
    OUTPUT_DIR / "alayna_small_cnn.pt",
)


def predict(raw_signals):
    with torch.no_grad():
        return model(to_tensor(raw_signals).to(DEVICE)).argmax(dim=1).cpu().numpy()


clean_predictions = predict(X_test)
print(f"Model parameters: {num_parameters:,}")
print(f"Best validation epoch: {best_epoch}")
print(f"Clean test accuracy: {accuracy_score(y_test, clean_predictions):.2%}")
print(classification_report(y_test, clean_predictions, target_names=CLASSES, digits=3, zero_division=0))
print(pd.DataFrame(confusion_matrix(y_test, clean_predictions), index=CLASSES, columns=CLASSES))

records = []


def save_scores(level_label, trial, y_pred):
    recalls = recall_score(y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    records.append({
        "noise_level": level_label,
        "trial": trial,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "normal_recall": float(recalls[0]),
        "inner_recall": float(recalls[1]),
        "outer_recall": float(recalls[2]),
    })


save_scores("Clean", 0, clean_predictions)
rng = np.random.default_rng(2026)
segment_rms = np.sqrt(np.mean(X_test.astype(np.float64) ** 2, axis=1, keepdims=True))

for snr_db in NOISE_LEVELS_DB:
    noise_rms = segment_rms / (10 ** (snr_db / 20))
    for trial in range(REPEATS):
        gaussian_noise = rng.standard_normal(X_test.shape, dtype=np.float32)
        noisy_test = X_test + gaussian_noise * noise_rms.astype(np.float32)
        save_scores(f"{snr_db} dB", trial, predict(noisy_test))

trials = pd.DataFrame.from_records(records)
level_order = ["Clean"] + [f"{v} dB" for v in NOISE_LEVELS_DB]
summary = (
    trials.groupby("noise_level", sort=False)
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        normal_recall=("normal_recall", "mean"),
        inner_recall=("inner_recall", "mean"),
        outer_recall=("outer_recall", "mean"),
    )
    .reindex(level_order)
)
summary["accuracy_sd"] = summary["accuracy_sd"].fillna(0.0)
trials.to_csv(OUTPUT_DIR / "noise_robustness_trials.csv", index=False)
summary.to_csv(OUTPUT_DIR / "noise_robustness_summary.csv")
print((summary * 100).round(1))
