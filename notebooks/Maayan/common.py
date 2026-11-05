import os
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

CLASS_NAMES = ["InnerRaceFault", "Normal", "OuterRaceFault"]


class PrepData:
    def data_to_torch(self):
        """convert np arrays into tensors"""
        data = np.load("data/standardized.npz")
        train_data = data["train_data"]
        val_data = data["val_data"]
        test_data = data["test_data"]

        train_t = torch.from_numpy(train_data.reshape(-1, 1, 5000)).float()
        val_t = torch.from_numpy(val_data.reshape(-1, 1, 5000)).float()
        test_t = torch.from_numpy(test_data.reshape(-1, 1, 5000)).float()

        return train_t, val_t, test_t

    def encode_labels(self):
        data = np.load("data/standardized.npz")

        train_labels = data["train_labels"]
        val_labels = data["val_labels"]
        test_labels = data["test_labels"]

        encoder = LabelEncoder()

        train_l = encoder.fit_transform(train_labels)
        val_l = encoder.transform(val_labels)
        test_l = encoder.transform(test_labels)

        train_label_tensor = torch.tensor(train_l, dtype=torch.long)
        val_label_tensor = torch.tensor(val_l, dtype=torch.long)
        test_label_tensor = torch.tensor(test_l, dtype=torch.long)

        return train_label_tensor, val_label_tensor, test_label_tensor

    def create_loaders(self):
        (
            train_data_tensor,
            val_data_tensor,
            test_data_tensor,
        ) = self.data_to_torch()
        train_label_tensor, val_label_tensor, test_label_tensor = self.encode_labels()

        train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
        val_dataset = TensorDataset(val_data_tensor, val_label_tensor)
        test_dataset = TensorDataset(test_data_tensor, test_label_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)

        return train_loader, val_loader, test_loader


class BaseFaultModel(nn.Module, ABC):
    """Shared base for the fault-classification CNNs.

    Subclasses define the architecture (layers in __init__ and forward) and may
    override the optimizer hyperparameters below.
    """

    num_classes = 3
    learning_rate = 1e-3
    weight_decay = 1e-4

    @abstractmethod
    def forward(self, x):
        """x: (batch, 1, 5000) -> logits: (batch, num_classes)"""

    def loss_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        return criterion, optimizer

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def training_loop(
    model, train_loader, val_loader, criterion, optimizer, checkpoint_path, num_epochs=50
):
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch+1}: loss = {running_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

        val_accuracy = correct / total
        print(f"val_loss = {val_loss / len(val_loader):.4f}, val_accuracy = {val_accuracy:.4f}")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)


def test_model(model, test_loader, checkpoint_path, confusion_matrix_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch_labels.tolist())

    test_accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"test_accuracy = {test_accuracy:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.savefig(confusion_matrix_path)
    plt.close()


def benchmark_model(model, test_loader, checkpoint_path, latency_runs=200):
    """Evaluate a model on the test set and report quality and size metrics.

    Returns a dict with accuracy, macro and per-class precision/recall/F1,
    total and non-zero param counts, dtype, size on disk, and median CPU
    latency for a single sample.
    """
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            all_predictions.extend(outputs.argmax(dim=1).tolist())
            all_labels.extend(batch_labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, labels=range(len(CLASS_NAMES)), zero_division=0
    )

    sample = next(iter(test_loader))[0][:1]
    with torch.no_grad():
        for _ in range(20):  # warmup
            model(sample)
        times = []
        for _ in range(latency_runs):
            start = time.perf_counter()
            model(sample)
            times.append(time.perf_counter() - start)

    params = list(model.parameters())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "per_class": {
            name: {"precision": p, "recall": r, "f1": f}
            for name, p, r, f in zip(CLASS_NAMES, precision, recall, f1)
        },
        "params": sum(p.numel() for p in params),
        "nonzero_params": sum(int((p != 0).sum().item()) for p in params),
        "dtype": str(params[0].dtype) if params else "int8 (quantized)",
        "size_kb": os.path.getsize(checkpoint_path) / 1024,
        "latency_ms": float(np.median(times)) * 1000,
    }


def save_confusion_matrix(model, test_loader, path):
    """Plot and save the test-set confusion matrix for an already-loaded model."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            all_predictions.extend(model(batch_data).argmax(dim=1).tolist())
            all_labels.extend(batch_labels.tolist())

    cm = confusion_matrix(all_labels, all_predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(cmap="Blues")
    plt.savefig(path)
    plt.close()
