# Shared helpers for training, evaluating and measuring the models.
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
    # train a fresh model per seed; returns test accuracies and one model per seed
    test_accs, models = [], []
    for seed in seeds:
        print(f"starting seed {seed}...", flush=True)
        start = time.perf_counter()
        torch.manual_seed(seed)
        model = model_class()
        val_acc = train_model(model, train_loader, val_loader, epochs, lr)
        test_accs.append(get_accuracy(model, test_loader))
        models.append(model)
        secs = time.perf_counter() - start
        print(f"seed {seed}: val {val_acc:.3f}, test {test_accs[-1]:.3f} ({secs:.0f} s)", flush=True)
    return test_accs, models


def ms_per_sample(model, sample, warmup=20, runs=500):
    model.eval()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            times.append(time.perf_counter() - start)
    torch.set_num_threads(old_threads)
    return sorted(times)[runs // 2] * 1000


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def flash_kb(model, bytes_per_param=4):
    return count_params(model) * bytes_per_param / 1024


def ram_kb(model, input_shape, bytes_per_value=4):
    counted = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)
    layers = list(model.features) + [model.classifier]
    sizes = []
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        for layer in layers:
            if isinstance(layer, nn.Linear):
                x = x.flatten(1)
            out = layer(x)
            if isinstance(layer, counted):
                sizes.append(x.numel() + out.numel())
            x = out
    return max(sizes) * bytes_per_value / 1024
