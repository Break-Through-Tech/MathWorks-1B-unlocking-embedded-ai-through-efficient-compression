# Sweeps pruning strength to see where accuracy breaks. Saves results/results_sweep.json
import torch, torch.nn as nn, torch.nn.utils.prune as prune
import json
import numpy as np
from pathlib import Path
from model import SpecCNN
from common import make_loader, get_accuracy, train_model, count_params, count_nonzero
from prune_common import shrink, keep_largest

HERE = Path(__file__).resolve().parent
BATCH_SIZE = 32
SEEDS = [0, 1, 2, 3, 4]
UNSTRUCTURED_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
STRUCTURED_CONFIGS = [(8, 16), (6, 12), (4, 8), (3, 6), (2, 4), (1, 2), (1, 1)]  # filters in conv1, conv2
FT_EPOCHS, FT_LR = 10, 0.0005

data = np.load(HERE / "processed_data" / "spectrograms_128.npz")
train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)

def load_baseline(seed):
    torch.manual_seed(seed)
    m = SpecCNN()
    m.load_state_dict(torch.load(HERE / "weights" / f"spec_cnn_baseline_seed{seed}.pt"))
    return m

results = {"unstructured": [], "structured": []}
for level in UNSTRUCTURED_LEVELS:
    accs, nz = [], 0
    for seed in SEEDS:
        model = load_baseline(seed)
        parts = [(l, "weight") for l in model.modules() if isinstance(l, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(parts, pruning_method=prune.L1Unstructured, amount=level)
        train_model(model, train_loader, val_loader, FT_EPOCHS, FT_LR)  # masks stay on, so pruned weights stay zero
        for l, w in parts:
            prune.remove(l, w)
        accs.append(get_accuracy(model, test_loader)); nz = count_nonzero(model)
    results["unstructured"].append({"level": level, "test_accs": accs, "nonzero_params": nz})
    print(f"unstructured {level:.0%}: {np.mean(accs):.3f} +- {np.std(accs):.3f}  nonzero {nz}", flush=True)

for f1, f2 in STRUCTURED_CONFIGS:
    accs, params = [], 0
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = load_baseline(seed)
        if (f1, f2) != (8, 16):
            model = shrink(model, "conv1", keep_largest(model, "conv1", f1))
            model = shrink(model, "conv2", keep_largest(model, "conv2", f2))
            train_model(model, train_loader, val_loader, 30, FT_LR)
        accs.append(get_accuracy(model, test_loader)); params = count_params(model)
    results["structured"].append({"filters": [f1, f2], "test_accs": accs, "params": params})
    print(f"structured {f1}/{f2}: {np.mean(accs):.3f} +- {np.std(accs):.3f}  params {params}", flush=True)

(HERE / "results" / "results_sweep.json").write_text(json.dumps(results))
