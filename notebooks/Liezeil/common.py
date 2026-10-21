# Shared helpers for training and evaluating the models.
import torch
import torch.nn as nn
import time
from torch.utils.data import TensorDataset, DataLoader


def predict(model, loader):
    # eval mode turns off training-only behavior (like BatchNorm updates)
    model.eval()
    guesses, truth = [], []
    with torch.no_grad():
        for X, y in loader:
            guesses.append(model(X).argmax(dim=1))
            truth.append(y)
    return torch.cat(guesses), torch.cat(truth)


def get_accuracy(model, loader):
    guesses, truth = predict(model, loader)
    return (guesses == truth).float().mean().item()


def train_model(model, train_loader, val_loader, epochs, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(X), y).backward()
            optimizer.step()
    return get_accuracy(model, val_loader)


def make_loader(X, y, batch_size, shuffle):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # add channel dim
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def run_seeds(model_class, train_loader, val_loader, test_loader, seeds, epochs, lr):
    # train a fresh model per seed; returns test accuracies and the last model
    test_accs = []
    for seed in seeds:
        print(f"starting seed {seed}...", flush=True)
        start = time.perf_counter()
        torch.manual_seed(seed)
        model = model_class()
        val_acc = train_model(model, train_loader, val_loader, epochs, lr)
        test_accs.append(get_accuracy(model, test_loader))
        secs = time.perf_counter() - start
        print(f"seed {seed}: val {val_acc:.3f}, test {test_accs[-1]:.3f} ({secs:.0f} s)", flush=True)
    return test_accs, model


def ms_per_sample(model, sample, warmup=20, runs=200):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):  # warm-up so the first slow runs don't count
            model(sample)
        start = time.perf_counter()
        for _ in range(runs):
            model(sample)
    return (time.perf_counter() - start) / runs * 1000