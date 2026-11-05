import torch
from pathlib import Path
from model import SpecCNN
from prune_common import shrink, keep_largest
from common import count_params

HERE = Path(__file__).resolve().parent

WEIGHTS_FILE = "spec_cnn_baseline_seed0.pt"
INPUT_SHAPE = (1, 65, 77)
CASES = [("conv1", 4), ("conv2", 8)]

model = SpecCNN()
model.load_state_dict(torch.load(HERE / "weights" / WEIGHTS_FILE))
model.eval()
x = torch.randn(8, *INPUT_SHAPE)

for layer, count in CASES:
    keep = keep_largest(model, layer, count)
    small = shrink(model, layer, keep)
    small.eval()

    masked = SpecCNN()
    masked.load_state_dict(model.state_dict())
    masked.eval()
    pos = {"conv1": 4, "conv2": 8}[layer]
    drop = [i for i in range(masked.features[pos - 4].out_channels) if i not in keep]
    masked.features[pos].weight.data[:, drop] = 0

    diff = (small(x) - masked(x)).abs().max().item()
    print(layer, "keep", count, "params", count_params(small), "max diff", diff)
