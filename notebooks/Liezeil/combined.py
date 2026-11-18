import torch
import json
import numpy as np
from pathlib import Path
from model import SpecCNN
from common import make_loader, get_accuracy, train_model, ms_per_sample, count_params, flash_kb, ram_kb
from projection_common import project_model
from quant_common import quantize

HERE = Path(__file__).resolve().parent

DATA_FILE = "spectrograms_128.npz"
RESULTS_FILE = "results_combined.json"
BATCH_SIZE = 32
SEEDS = [0, 1, 2, 3, 4]
INPUT_SHAPE = (1, 65, 77)
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 0.0005
VARIANTS = [
    ("c1_4_c2_4_r3_4", 4, 4, {"conv3": 4}),
    ("c1_4_c2_4_r3_2", 4, 4, {"conv3": 2}),
    ("c1_4_c2_4_r2_2_r3_2", 4, 4, {"conv2": 2, "conv3": 2}),
    ("c1_2_c2_4_r3_2", 2, 4, {"conv3": 2}),
    ("c1_2_c2_4_r2_2_r3_2", 2, 4, {"conv2": 2, "conv3": 2}),
]

data = np.load(HERE / "processed_data" / DATA_FILE)
train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
calib_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=False)
val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)

results = {}
for name, f1, f2, ranks in VARIANTS:
    fp32_accs, int8_accs = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SpecCNN(conv1_filters=f1, conv2_filters=f2)
        model.load_state_dict(torch.load(HERE / "weights" / f"spec_cnn_c1_{f1}_c2_{f2}_seed{seed}.pt"))
        model = project_model(model, ranks)
        train_model(model, train_loader, val_loader, FINE_TUNE_EPOCHS, FINE_TUNE_LR)
        model.eval()
        quant_model = quantize(model, calib_loader)
        fp32_accs.append(get_accuracy(model, test_loader))
        int8_accs.append(get_accuracy(quant_model, test_loader))
    sample = torch.zeros(1, *INPUT_SHAPE)
    results[name] = {
        "fp32_accs": fp32_accs,
        "int8_accs": int8_accs,
        "params": count_params(model),
        "flash_kb_int8": flash_kb(model, bytes_per_param=1),
        "ram_kb_int8": ram_kb(model, INPUT_SHAPE, bytes_per_value=1),
        "ms_fp32": ms_per_sample(model, sample),
        "ms_int8": ms_per_sample(quant_model, sample),
    }
    r = results[name]
    print(f"{name:22} fp32 {np.mean(fp32_accs):.3f}  int8 {np.mean(int8_accs):.3f} +- {np.std(int8_accs):.3f}  "
          f"params {r['params']:,}  flash {r['flash_kb_int8']:.2f} KB  ram {r['ram_kb_int8']:.1f} KB  "
          f"ms fp32 {r['ms_fp32']:.3f} int8 {r['ms_int8']:.3f}", flush=True)

(HERE / "results" / RESULTS_FILE).write_text(json.dumps(results))
