import copy

import torch
import torch.ao.quantization as tq

from notebooks.Maayan.common import PrepData, benchmark_model, save_confusion_matrix
from notebooks.Maayan.largeCNN.train_large_cnn import FaultCNNLarge, CHECKPOINT_PATH

INT8_CHECKPOINT_PATH = "notebooks/Maayan/largeCNN/models/large_model_int8.pt"
INT8_CONFUSION_MATRIX_PATH = "notebooks/Maayan/largeCNN/results/confusion_matrix_int8.png"
CALIBRATION_BATCHES = 20


def fuse(model):
    """Fuse each Conv+BN+ReLU block and the classifier's Linear+ReLU in place."""
    for i in range(len(model.features)):
        tq.fuse_modules(model.features[i], ["0", "1", "2"], inplace=True)
    tq.fuse_modules(model.classifier, ["0", "1"], inplace=True)
    return model


def quantize(model, train_loader):
    torch.backends.quantized.engine = "qnnpack"

    model = fuse(copy.deepcopy(model).eval())
    # BN fusion keeps the same params minus BN's, so count them before conversion
    # replaces the weights with packed int8 params
    params = model.count_params()
    wrapped = tq.QuantWrapper(model)
    wrapped.qconfig = tq.get_default_qconfig("qnnpack")
    tq.prepare(wrapped, inplace=True)

    # calibration: run training batches through to record activation ranges
    with torch.no_grad():
        for i, (batch_data, _) in enumerate(train_loader):
            if i >= CALIBRATION_BATCHES:
                break
            wrapped(batch_data)

    tq.convert(wrapped, inplace=True)
    return wrapped, params


if __name__ == "__main__":
    # run from the repo root: python -m notebooks.Maayan.largeCNN.quantize_large_cnn
    train_loader, _, test_loader = PrepData().create_loaders()

    model = FaultCNNLarge()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    quantized, params = quantize(model, train_loader)
    torch.save(quantized.state_dict(), INT8_CHECKPOINT_PATH)

    save_confusion_matrix(quantized, test_loader, INT8_CONFUSION_MATRIX_PATH)
    results = benchmark_model(quantized, test_loader, INT8_CHECKPOINT_PATH)
    # quantized layers hide their weights in packed params, so benchmark_model
    # can't count them; use the count taken before conversion
    results["params"] = params
    results["nonzero_params"] = params

    for key, value in results.items():
        if key == "per_class":
            for name, scores in value.items():
                print(f"  {name}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
        else:
            print(key, value if isinstance(value, (int, str)) else round(float(value), 4))
