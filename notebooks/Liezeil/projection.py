import copy
import torch
import json
import numpy as np
from pathlib import Path
from model import SpecCNN
from common import make_loader, get_accuracy, train_model, ms_per_sample, count_params, flash_kb, ram_kb
from projection_common import project_model

HERE = Path(__file__).resolve().parent

DATA_FILE = "spectrograms_128.npz"
WEIGHTS_PREFIX = "spec_cnn_baseline_seed"
RESULTS_FILE = "results_projection.json"
BATCH_SIZE = 32
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [(8, 16), (4, 8), (4, 4), (2, 4), (2, 2)]  # (conv2 rank, conv3 rank)
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 0.0005
INPUT_SHAPE = (1, 65, 77)

data = np.load(HERE / "processed_data" / DATA_FILE)
train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)

results = {}
for r2, r3 in CONFIGS:
    name = f"r2_{r2}_r3_{r3}"
    before_accs, test_accs = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SpecCNN()
        model.load_state_dict(torch.load(HERE / "weights" / f"{WEIGHTS_PREFIX}{seed}.pt"))
        model = project_model(model, {"conv2": r2, "conv3": r3})
        before_accs.append(get_accuracy(model, test_loader))
        train_model(model, train_loader, val_loader, FINE_TUNE_EPOCHS, FINE_TUNE_LR)
        test_accs.append(get_accuracy(model, test_loader))
    results[name] = {
        "acc_before_finetune": before_accs,
        "test_accs": test_accs,
        "params": count_params(model),
        "flash_kb": flash_kb(model),
        "ram_kb": ram_kb(model, INPUT_SHAPE),
        "ms": ms_per_sample(model, torch.zeros(1, *INPUT_SHAPE)),
    }
    print(f"{name:12} before {np.mean(before_accs):.3f}  test {np.mean(test_accs):.3f} +- {np.std(test_accs):.3f}  "
          f"params {results[name]['params']:,}  flash {results[name]['flash_kb']:.1f} KB  "
          f"ram {results[name]['ram_kb']:.1f} KB  {results[name]['ms']:.3f} ms", flush=True)

(HERE / "results" / RESULTS_FILE).write_text(json.dumps(results))
