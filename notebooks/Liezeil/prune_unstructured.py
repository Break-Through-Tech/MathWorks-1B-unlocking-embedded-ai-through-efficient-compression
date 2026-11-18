import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import json
import numpy as np
from pathlib import Path
from model import SpecCNN
from common import make_loader, get_accuracy, train_model, ms_per_sample, count_params, count_nonzero, flash_kb, ram_kb

HERE = Path(__file__).resolve().parent

DATA_FILE = "spectrograms_128.npz"
WEIGHTS_PREFIX = "spec_cnn_baseline_seed"
RESULTS_FILE = "results_unstructured.json"
BATCH_SIZE = 32
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = [("unstructured_50", 0.5, False), ("unstructured_50_ft", 0.5, True), ("unstructured_80_ft", 0.8, True)]
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0005
INPUT_SHAPE = (1, 65, 77)

data = np.load(HERE / "processed_data" / DATA_FILE)
train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)

results = {}
for name, level, fine_tune in VARIANTS:
    test_accs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SpecCNN()
        model.load_state_dict(torch.load(HERE / "weights" / f"{WEIGHTS_PREFIX}{seed}.pt"))
        parts = []
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                parts.append((layer, "weight"))
        prune.global_unstructured(parts, pruning_method=prune.L1Unstructured, amount=level)
        if fine_tune:
            train_model(model, train_loader, val_loader, FINE_TUNE_EPOCHS, FINE_TUNE_LR)
        for layer, weight_name in parts:
            prune.remove(layer, weight_name)
        test_accs.append(get_accuracy(model, test_loader))
    results[name] = {
        "test_accs": test_accs,
        "params": count_params(model),
        "nonzero_params": count_nonzero(model),
        "flash_kb": flash_kb(model),
        "ram_kb": ram_kb(model, INPUT_SHAPE),
        "ms": ms_per_sample(model, torch.zeros(1, *INPUT_SHAPE)),
    }
    r = results[name]
    print(f"{name:20} test {np.mean(test_accs):.3f} +- {np.std(test_accs):.3f}  non-zero {r['nonzero_params']:,} of {r['params']:,}", flush=True)

(HERE / "results" / RESULTS_FILE).write_text(json.dumps(results))
