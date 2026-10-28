"""Train a compact 1D CNN on standardized bearing vibration signals.

In the mathworks1b environment, run:
    python standardize_data.py
    python train_model.py
    python train_model.py --epochs 50 --batch-size 32 --learning-rate 0.001

Each run saves its best checkpoint, learning history, and test metrics in a
new outputs/training/<timestamp>/ directory. Inputs must already be standardized.
"""

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["Normal", "InnerRaceFault", "OuterRaceFault"]


class BearingCNN(nn.Module):
    """Input: (batch, 1, signal_length). Output: three unnormalized logits."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, signals):
        return self.network(signals)


def load_split(data_dir, split):
    path = data_dir / f"{split}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run standardize_data.py first.")
    contents = loadmat(path)
    signals = np.asarray(contents[f"{split}Data"], dtype=np.float32)
    labels = [str(label.item()) for label in contents[f"{split}Labels"].ravel()]
    if signals.ndim != 2 or signals.shape[0] == 0 or signals.shape[1] < 64:
        raise ValueError(f"{split}: expected a nonempty matrix with length >= 64.")
    if not np.isfinite(signals).all() or len(labels) != len(signals):
        raise ValueError(f"{split}: nonfinite signals or mismatched labels.")
    unknown = set(labels) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"{split}: unknown classes: {sorted(unknown)}")
    targets = torch.tensor([CLASS_NAMES.index(label) for label in labels])
    # Conv1d expects a channel dimension between batch and time.
    return TensorDataset(torch.from_numpy(signals).unsqueeze(1), targets)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    actual, predicted = [], []
    with torch.set_grad_enabled(training):
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; check input data and learning rate.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
    accuracy = float(np.mean(np.array(actual) == np.array(predicted)))
    return total_loss / len(actual), accuracy, actual, predicted


def print_confusion_matrix(matrix, class_names):
    """Print counts with actual classes in rows and predicted classes in columns."""
    width = max(len(name) for name in class_names) + 2
    print("\nTest confusion matrix (counts)")
    print("Rows = actual class; columns = predicted class\n")
    print(f"{'Actual / Predicted':>{width}}" + "".join(
        f"{name:>{width}}" for name in class_names))
    for name, row in zip(class_names, matrix):
        print(f"{name:>{width}}" + "".join(f"{int(value):>{width}}" for value in row))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1:
        parser.error("epochs, batch-size, patience, and threads must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = PROJECT_DIR / "data" / "standardized"
    datasets = {split: load_split(data_dir, split) for split in ("train", "val", "test")}
    signal_length = datasets["train"].tensors[0].shape[-1]
    if any(ds.tensors[0].shape[-1] != signal_length for ds in datasets.values()):
        raise ValueError("All splits must have the same signal length.")
    if set(datasets["train"].tensors[1].tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("Training data must contain every class.")
    stats = loadmat(data_dir / "standardization.mat")
    mean, std = stats["training_mean"].item(), stats["training_std"].item()
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid saved standardization statistics.")
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    output_dir = PROJECT_DIR / "outputs" / "training" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)
    model = BearingCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}; parameters: {parameter_count:,}", flush=True)
    best_loss, stale_epochs = float("inf"), 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, loaders["val"], criterion, device)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                            val_loss=val_loss, val_accuracy=val_acc))
        print(f"Epoch {epoch:02d}: train loss={train_loss:.4f}, acc={train_acc:.1%}; "
              f"val loss={val_loss:.4f}, acc={val_acc:.1%}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            # Includes everything needed to reconstruct the model and preprocess
            # raw signals. Load state into BearingCNN; do not standardize twice.
            torch.save({
                "model_state_dict": {k: v.cpu() for k, v in best_state.items()},
                "architecture": "BearingCNN", "class_names": CLASS_NAMES,
                "signal_length": signal_length, "training_mean": mean,
                "training_std": std, "best_epoch": best_epoch,
                "val_loss": best_loss, "arguments": vars(args),
            }, output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation loss stopped improving.")
                break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # Test only after selecting the best epoch using validation loss.
    model.load_state_dict(best_state)
    test_loss, test_acc, actual, predicted = run_epoch(model, loaders["test"], criterion, device)
    class_ids = list(range(len(CLASS_NAMES)))
    metrics = {
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "test_loss": test_loss, "test_accuracy": test_acc,
        "parameter_count": parameter_count, "class_names": CLASS_NAMES,
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(
            actual, predicted, labels=class_ids).tolist(),
        "classification_report": classification_report(
            actual, predicted, labels=class_ids, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0),
        "arguments": vars(args), "device": str(device),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best epoch: {best_epoch}; test accuracy: {test_acc:.1%}")
    print(classification_report(actual, predicted, labels=class_ids,
                                target_names=CLASS_NAMES, zero_division=0))
    print_confusion_matrix(
        metrics["confusion_matrix_rows_actual_columns_predicted"], CLASS_NAMES)
    print(f"Saved checkpoint, history, and metrics to: {output_dir}")


if __name__ == "__main__":
    main()
