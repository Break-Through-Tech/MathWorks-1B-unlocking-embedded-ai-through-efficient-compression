from abc import ABC, abstractmethod

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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
