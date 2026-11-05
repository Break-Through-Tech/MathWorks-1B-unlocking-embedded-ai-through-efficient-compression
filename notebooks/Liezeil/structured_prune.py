import torch
import json
import numpy as np
from pathlib import Path
from model import SpecCNN
from common import make_loader, get_accuracy, train_model, ms_per_sample, count_params, count_nonzero, flash_kb, ram_kb
from prune_common import shrink, keep_largest

HERE = Path(__file__).resolve().parent

DATA_FILE = "spectrograms_128.npz"
WEIGHTS_PREFIX = "spec_cnn_baseline_seed"
RESULTS_FILE = "results_structured.json"
BATCH_SIZE = 32
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [(4, 16), (2, 16), (8, 8), (4, 8), (4, 4), (2, 4)]
SAVE_CONFIGS = [(4, 4), (2, 4)]
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 0.0005
INPUT_SHAPE = (1, 65, 77)

data = np.load(HERE / "processed_data" / DATA_FILE)
train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)

results = {}
for filters1, filters2 in CONFIGS:
    name = f"c1_{filters1}_c2_{filters2}"
    test_accs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SpecCNN()
        model.load_state_dict(torch.load(HERE / "weights" / f"{WEIGHTS_PREFIX}{seed}.pt"))
        model = shrink(model, "conv1", keep_largest(model, "conv1", filters1))
        model = shrink(model, "conv2", keep_largest(model, "conv2", filters2))
        train_model(model, train_loader, val_loader, FINE_TUNE_EPOCHS, FINE_TUNE_LR)
        test_accs.append(get_accuracy(model, test_loader))
        if (filters1, filters2) in SAVE_CONFIGS:
            torch.save(model.state_dict(), HERE / "weights" / f"spec_cnn_{name}_seed{seed}.pt")
    results[name] = {
        "test_accs": test_accs,
        "params": count_params(model),
        "nonzero_params": count_nonzero(model),
        "flash_kb": flash_kb(model),
        "ram_kb": ram_kb(model, INPUT_SHAPE),
        "ms": ms_per_sample(model, torch.zeros(1, *INPUT_SHAPE)),
    }
    print(f"{name:12} test {np.mean(test_accs):.3f} +- {np.std(test_accs):.3f}  "
          f"params {results[name]['params']:,}  flash {results[name]['flash_kb']:.1f} KB  "
          f"ram {results[name]['ram_kb']:.1f} KB  {results[name]['ms']:.3f} ms", flush=True)

(HERE / "results" / RESULTS_FILE).write_text(json.dumps(results))
