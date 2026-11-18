import torch

from notebooks.Maayan.common import PrepData, benchmark_model, save_confusion_matrix
from notebooks.Maayan.largeCNN.quantize_large_cnn import quantize
from notebooks.Maayan.largeCNN.train_large_cnn import FaultCNNLarge
from notebooks.Maayan.prune_structured_large_cnn import PRUNED_CHECKPOINT_PATH

WIDTHS = (8, 16, 32, 64)  # the KEEP = 0.5 model
INT8_CHECKPOINT_PATH = "notebooks/Maayan/largeCNN/models/large_model_structured_int8.pt"
INT8_CONFUSION_MATRIX_PATH = "notebooks/Maayan/largeCNN/results/confusion_matrix_structured_int8.png"


if __name__ == "__main__":
    # run from the repo root: python -m notebooks.Maayan.quantize_structured_large_cnn
    train_loader, _, test_loader = PrepData().create_loaders()

    model = FaultCNNLarge(widths=WIDTHS)
    model.load_state_dict(torch.load(PRUNED_CHECKPOINT_PATH))

    quantized, params = quantize(model, train_loader)
    torch.save(quantized.state_dict(), INT8_CHECKPOINT_PATH)

    save_confusion_matrix(quantized, test_loader, INT8_CONFUSION_MATRIX_PATH)
    for i in range(3):
        results = benchmark_model(quantized, test_loader, INT8_CHECKPOINT_PATH)
        print("latency run", i, round(results["latency_ms"], 3))
    results["params"] = params
    results["nonzero_params"] = params

    for key, value in results.items():
        if key == "per_class":
            for name, scores in value.items():
                print(f"  {name}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
        else:
            print(key, value if isinstance(value, (int, str)) else round(float(value), 4))
