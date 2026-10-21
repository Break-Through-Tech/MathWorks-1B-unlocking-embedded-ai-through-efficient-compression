# Compares the 1D CNN and spectrogram CNN on size, accuracy, and speed
import torch
from pathlib import Path
import json
import numpy as np
from model import CNN, SpecCNN
from common import ms_per_sample

HERE = Path(__file__).resolve().parent

# name, class, results file, shape of one input
MODELS = [("1D CNN", CNN, "results_1d.json", (1, 1, 5000)),
          ("Spectrogram CNN", SpecCNN, "results_spec.json", (1, 1, 65, 77))]

print(f"{'model':18} {'params':>8} {'flash KB':>9} {'mean acc':>9} {'std':>7} {'ms/sample':>10}")
for name, model_class, results_file, shape in MODELS:
    accs = json.loads((HERE / results_file).read_text())["test_accs"]
    model = model_class()
    params = sum(p.numel() for p in model.parameters())
    flash_kb = params * 4 / 1024
    ms = ms_per_sample(model, torch.zeros(shape))
    print(f"{name:18} {params:>8,} {flash_kb:>9.1f} {np.mean(accs):>9.3f} {np.std(accs):>7.3f} {ms:>10.3f}")