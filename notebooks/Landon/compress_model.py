"""Prune and quantize the trained BearingCNN and compare accuracy, size, and speed.

In the mathworks1b environment, after train_model.py has saved a checkpoint, run:
    python compress_model.py
    python compress_model.py --checkpoint outputs/training/<run>/best_model.pt

Each run saves results.csv, results.json, compression_tradeoffs.png, and every
compressed model (TorchScript) in a new outputs/compression/<timestamp>/ directory.
"""

import argparse
import copy
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import random
import statistics
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
import torch.ao.nn.quantized as nnq
import torch.ao.quantization as tq
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from train_model import PROJECT_DIR, BearingCNN, load_split, run_epoch


CPU = torch.device("cpu")
KB = 1024
CSV_COLUMNS = ["model", "technique", "channels", "sparsity", "test_accuracy", "test_loss",
               "val_accuracy", "val_loss", "val_check", "nonzero_parameters", "parameters",
               "macs", "weight_kb", "compression_ratio", "gzip_kb", "latency_ms"]


def choose_engine(requested):
    supported = torch.backends.quantized.supported_engines
    for engine in [requested] if requested else ["fbgemm", "x86", "qnnpack"]:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No usable INT8 engine; this PyTorch build supports {supported}.")


def find_checkpoint():
    """Training run with the lowest validation loss."""
    runs = []
    for metrics_path in sorted((PROJECT_DIR / "outputs" / "training").glob("*/metrics.json"),
                               reverse=True):
        checkpoint = metrics_path.with_name("best_model.pt")
        if checkpoint.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            runs.append((metrics["best_val_loss"], checkpoint))
    if not runs:
        raise FileNotFoundError("No checkpoint in outputs/training/. Run train_model.py first.")
    return min(runs, key=lambda run: run[0])[1]


def evaluate(model, dataset):
    signals, labels = dataset.tensors
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch) for batch in signals.split(64)])
    accuracy = float((logits.argmax(dim=1) == labels).float().mean())
    return accuracy, nn.functional.cross_entropy(logits, labels).item()


def fine_tune(model, loaders, args):
    """Retrain after pruning and keep the epoch with the lowest validation loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = run_epoch(model, loaders["val"], criterion, CPU)[0]
    best_state, stale_epochs = copy.deepcopy(model.state_dict()), 0
    for _ in range(args.finetune_epochs):
        run_epoch(model, loaders["train"], criterion, CPU, optimizer)
        val_loss = run_epoch(model, loaders["val"], criterion, CPU)[0]
        if val_loss < best_loss:
            best_loss, best_state, stale_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def prunable(model):
    return [(layer, "weight") for layer in model.modules()
            if isinstance(layer, (nn.Conv1d, nn.Linear))]


def prune_unstructured(model, sparsity):
    """Zero the smallest weights across all layers until `sparsity` of them are zero."""
    targets = prunable(model)
    total = sum(layer.weight.numel() for layer, _ in targets)
    zeros = sum(int((layer.weight_mask == 0).sum()) for layer, _ in targets
                if hasattr(layer, "weight_mask"))
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured,
                              amount=max(0, round(sparsity * total) - zeros))


def remove_masks(model):
    plain = copy.deepcopy(model)
    for layer, name in prunable(plain):
        if hasattr(layer, name + "_orig"):
            prune.remove(layer, name)
    return plain.eval()


def conv_widths(model):
    return [layer.out_channels for layer in model.modules() if isinstance(layer, nn.Conv1d)]


def prune_structured(model, widths):
    """Keep the largest-L1 filters in each Conv1d and drop their inputs in the next layer."""
    slim = copy.deepcopy(model)
    layers, widths = slim.network, iter(widths)
    kept = previous_width = None
    for index, layer in enumerate(layers):
        if isinstance(layer, nn.Conv1d):
            weight = layer.weight.detach() if kept is None else layer.weight.detach()[:, kept]
            keep = weight.abs().sum(dim=(1, 2)).argsort(descending=True)[:next(widths)].sort().values
            smaller = nn.Conv1d(weight.shape[1], len(keep), layer.kernel_size, stride=layer.stride,
                                padding=layer.padding, dilation=layer.dilation)
            with torch.no_grad():
                smaller.weight.copy_(weight[keep])
                smaller.bias.copy_(layer.bias.detach()[keep])
            layers[index] = smaller
            kept, previous_width = keep, layer.out_channels
        elif isinstance(layer, nn.Linear):
            if layer.in_features != previous_width:
                raise ValueError("Expected global pooling between the last Conv1d and Linear layer.")
            smaller = nn.Linear(len(kept), layer.out_features)
            with torch.no_grad():
                smaller.weight.copy_(layer.weight.detach()[:, kept])
                smaller.bias.copy_(layer.bias.detach())
            layers[index] = smaller
    return slim.eval()


def quantize_int8(model, calibration_signals):
    """Post-training static INT8 quantization, calibrated on training signals."""
    layers = copy.deepcopy(model).eval().network
    conv_relu = [[str(i), str(i + 1)] for i in range(len(layers) - 1)
                 if isinstance(layers[i], nn.Conv1d) and isinstance(layers[i + 1], nn.ReLU)]
    tq.fuse_modules(layers, conv_relu, inplace=True)
    quantized = nn.Sequential(tq.QuantStub(), layers, tq.DeQuantStub()).eval()
    quantized.qconfig = tq.get_default_qconfig(torch.backends.quantized.engine)
    tq.prepare(quantized, inplace=True)
    with torch.no_grad():
        for batch in calibration_signals.split(64):
            quantized(batch)
    return tq.convert(quantized, inplace=True).eval()


def stored_arrays(model):
    """Every number the model stores: FP32 values, or INT8 weights plus FP32 biases and scales."""
    arrays = []
    for layer in model.modules():
        if isinstance(layer, nnq.Quantize):
            arrays += [layer.scale.numpy().astype(np.float32),
                       layer.zero_point.numpy().astype(np.int32)]
        elif isinstance(layer, (nnq.Conv1d, nnq.Linear)):
            weight, bias = layer.weight(), layer.bias()
            arrays.append(weight.int_repr().numpy())
            if weight.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                arrays += [weight.q_per_channel_scales().numpy().astype(np.float32),
                           weight.q_per_channel_zero_points().numpy().astype(np.int32)]
            else:
                arrays += [np.float32([weight.q_scale()]), np.int32([weight.q_zero_point()])]
            if bias is not None:
                arrays.append(bias.detach().numpy().astype(np.float32))
            arrays += [np.float32([layer.scale]), np.int32([layer.zero_point])]
        elif isinstance(layer, (nn.Conv1d, nn.Linear)):
            arrays += [p.detach().numpy() for p in layer.parameters(recurse=False)]
    return arrays


def count_parameters(model):
    nonzero = total = 0
    for layer in model.modules():
        if isinstance(layer, (nnq.Conv1d, nnq.Linear)):
            tensors = [layer.weight().int_repr(), layer.bias()]
        elif isinstance(layer, (nn.Conv1d, nn.Linear)):
            tensors = [layer.weight, layer.bias]
        else:
            continue
        for tensor in (t for t in tensors if t is not None):
            total += tensor.numel()
            nonzero += int(torch.count_nonzero(tensor))
    return nonzero, total


def count_macs(float_model, signal_length):
    """Multiply-accumulates to classify one signal."""
    macs = 0

    def count(layer, inputs, output):
        nonlocal macs
        if isinstance(layer, nn.Conv1d):
            macs += output.numel() * layer.in_channels // layer.groups * layer.kernel_size[0]
        else:
            macs += layer.in_features * layer.out_features

    hooks = [layer.register_forward_hook(count) for layer in float_model.modules()
             if isinstance(layer, (nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        float_model(torch.zeros(1, 1, signal_length))
    for hook in hooks:
        hook.remove()
    return macs


def latency_ms(model, signal, runs, blocks=5):
    """Single-thread time per signal: the fastest of several block medians."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for _ in range(max(10, runs // 10)):
                model(signal)
            medians = []
            for _ in range(blocks):
                times = []
                for _ in range(max(1, runs // blocks)):
                    start = time.perf_counter()
                    model(signal)
                    times.append(time.perf_counter() - start)
                medians.append(statistics.median(times))
    finally:
        torch.set_num_threads(threads)
    return 1000 * min(medians)


def smallest(rows):
    return min(rows, key=lambda row: (row["weight_kb"], row["nonzero_parameters"]))


def describe(row):
    if row["technique"] == "Unstructured pruning":
        return f"{row['sparsity']:.0%} of weights zeroed"
    return f"{row['channels']} channels"


def print_table(results):
    header = ["Model", "Test acc.", "Validation", "Nonzero / total params", "MACs",
              "Weights (KB)", "Compression", "Gzipped (KB)", "Latency (ms)"]
    rows = [[r["model"], f"{r['test_accuracy']:.1%}", r["val_check"],
             f"{r['nonzero_parameters']:,} / {r['parameters']:,}", f"{r['macs'] / 1000:,.0f}K",
             f"{r['weight_kb']:.2f}", f"{r['compression_ratio']:.1f}x", f"{r['gzip_kb']:.2f}",
             f"{r['latency_ms']:.3f}"] for r in results]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    for row in [header, ["-" * width for width in widths]] + rows:
        print("  ".join(cell.ljust(width) if i < 3 else cell.rjust(width)
                        for i, (cell, width) in enumerate(zip(row, widths))))


def plot_tradeoffs(path, results, best, recommended, max_val_loss):
    baseline = results[0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6),
                                      gridspec_kw={"width_ratios": [1, 1.25]})

    for technique, color, marker in [("Structured pruning", "#2a78d6", "o"),
                                     ("Unstructured pruning", "#eb6834", "s")]:
        rows = [baseline] + [r for r in results if r["technique"] == technique]
        left.plot([r["nonzero_parameters"] for r in rows], [100 * r["test_accuracy"] for r in rows],
                  color=color, marker=marker, linewidth=2, markersize=8, markeredgecolor="white",
                  markeredgewidth=2, label=technique + " (FP32)")
    counts = [r["nonzero_parameters"] for r in results]
    left.set_xscale("log")
    left.set_xlim(max(counts) * 1.4, min(counts) / 1.4)  # More pruning to the right.
    ticks = [t for t in (10000, 3000, 1000, 300, 100) if min(counts) / 1.4 <= t <= max(counts) * 1.4]
    left.set_xticks(ticks, [f"{t:,}" for t in ticks])
    left.xaxis.set_minor_formatter(NullFormatter())
    lowest = min(100 * r["test_accuracy"] for r in results)
    left.set_ylim(min(90, 5 * np.floor((lowest - 2) / 5)), 101.5)
    left.set(xlabel=f"Nonzero parameters, log scale (baseline {baseline['parameters']:,})",
             ylabel="Test accuracy (%)")
    left.set_title("Test accuracy as pruning removes parameters", loc="left")
    left.legend(frameon=False, loc="lower left")

    rows = [baseline] + [r for r in best if r is not baseline]
    positions = np.arange(len(rows))[::-1]
    right.barh(positions, [r["weight_kb"] for r in rows], height=0.5,
               color=["#2a78d6" if r is recommended else "#c3c2b7" for r in rows])
    for position, r in zip(positions, rows):
        label = f"{r['weight_kb']:.1f} KB"
        if r["technique"] == "Unstructured pruning":
            label += f" ({r['gzip_kb']:.1f} KB gzipped)"
        elif r is not baseline:
            label += f"  ·  {r['compression_ratio']:.1f}x smaller"
        right.text(r["weight_kb"] + 1, position, f"{label}  ·  {r['test_accuracy']:.0%} test acc.",
                   va="center", fontsize=9)
    right.set_yticks(positions, [f"{r['technique']}\n{describe(r)}"
                                 + ("  ·  recommended" if r is recommended else "") for r in rows],
                     fontsize=9)
    right.set_xlim(0, baseline["weight_kb"] * 1.85)
    right.set_xticks(np.arange(0, baseline["weight_kb"] + 1, 20))
    right.set_xlabel("Weight memory (KB)")
    right.set_title("Smallest model per technique that passed validation", loc="left")

    for ax, grid_axis in [(left, "y"), (right, "x")]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis=grid_axis, color="#e1e0d9")
        ax.set_axisbelow(True)
        ax.tick_params(which="both", length=0)
    fig.text(0.01, 0.015, "Passed validation = baseline validation accuracy and validation loss "
             f"<= {max_val_loss:.2f}. Test accuracy on held-out signals.", color="#898781",
             fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        help="default: the training run with the lowest validation loss")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95, 0.98],
                        help="unstructured pruning: fraction of weights set to zero")
    parser.add_argument("--keep-fractions", type=float, nargs="+",
                        default=[0.75, 0.5, 0.375, 0.25, 0.125],
                        help="structured pruning: fraction of filters kept per layer")
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-val-drop", type=float, default=0.0,
                        help="validation accuracy drop a model may have and still pass")
    parser.add_argument("--max-val-loss", type=float, default=0.1,
                        help="largest validation loss a model may have and still pass")
    parser.add_argument("--engine", choices=["fbgemm", "x86", "onednn", "qnnpack"],
                        help="INT8 backend (default: fbgemm, else x86, else qnnpack)")
    parser.add_argument("--latency-runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.finetune_epochs, args.patience, args.batch_size, args.latency_runs,
           args.threads) < 1:
        parser.error("epoch, patience, batch, latency-run, and thread counts must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.max_val_loss <= 0:
        parser.error("learning-rate and max-val-loss must be positive")
    if not 0 <= args.max_val_drop < 1:
        parser.error("max-val-drop must be in [0, 1)")
    if args.sparsities != sorted(set(args.sparsities)) or not all(0 < s < 1 for s in args.sparsities):
        parser.error("sparsities must be increasing values between 0 and 1")
    if (args.keep_fractions != sorted(set(args.keep_fractions), reverse=True)
            or not all(0 < k < 1 for k in args.keep_fractions)):
        parser.error("keep-fractions must be decreasing values between 0 and 1")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    engine = choose_engine(args.engine)
    warnings.filterwarnings("ignore", message="Please use quant_min and quant_max")

    checkpoint_path = (args.checkpoint or find_checkpoint()).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=CPU, weights_only=True)
    baseline = BearingCNN()
    baseline.load_state_dict(checkpoint["model_state_dict"])
    baseline.eval()
    print(f"Baseline: {checkpoint_path}\nINT8 engine: {engine}", flush=True)

    data_dir = PROJECT_DIR / "data" / "standardized"
    datasets = {split: load_split(data_dir, split) for split in ("train", "val", "test")}
    signal_length = datasets["train"].tensors[0].shape[-1]
    stats = loadmat(data_dir / "standardization.mat")
    if signal_length != checkpoint["signal_length"] or not np.allclose(
            [stats["training_mean"].item(), stats["training_std"].item()],
            [checkpoint["training_mean"], checkpoint["training_std"]]):
        raise ValueError("data/standardized does not match the data this checkpoint was trained on.")
    if evaluate(baseline, datasets["val"])[0] < 0.9:
        print("WARNING: this checkpoint is undertrained (under 90% validation accuracy); "
              "pass --checkpoint to use a better run.", flush=True)

    loaders = {"train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
               "val": DataLoader(datasets["val"], batch_size=args.batch_size)}
    calibration = datasets["train"].tensors[0]
    one_signal = datasets["test"].tensors[0][:1]
    output_dir = (PROJECT_DIR / "outputs" / "compression"
                  / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    (output_dir / "models").mkdir(parents=True)
    results = []

    def record(name, technique, model, float_model, sparsity=0.0):
        val_accuracy, val_loss = evaluate(model, datasets["val"])
        test_accuracy, test_loss = evaluate(model, datasets["test"])
        nonzero, total = count_parameters(model)
        stored = b"".join(np.ascontiguousarray(array).tobytes() for array in stored_arrays(model))
        torch.jit.save(torch.jit.script(model), str(output_dir / "models" / f"{name}.pt"))
        results.append({
            "model": name, "technique": technique,
            "channels": "-".join(map(str, conv_widths(float_model))), "sparsity": sparsity,
            "test_accuracy": test_accuracy, "test_loss": test_loss,
            "val_accuracy": val_accuracy, "val_loss": val_loss, "val_check": "",
            "nonzero_parameters": nonzero, "parameters": total,
            "macs": count_macs(float_model, signal_length),
            "weight_kb": len(stored) / KB,
            "compression_ratio": results[0]["weight_kb"] * KB / len(stored) if results else 1.0,
            "gzip_kb": len(gzip.compress(stored, mtime=0)) / KB,
            "latency_ms": latency_ms(model, one_signal, args.latency_runs),
        })
        print(f"  {name}: test acc {test_accuracy:.1%}, val loss {val_loss:.4f}, "
              f"{len(stored) / KB:.2f} KB", flush=True)

    record("baseline_fp32", "Baseline (FP32)", baseline, baseline)
    record("int8", "INT8 quantization", quantize_int8(baseline, calibration), baseline)

    masked = copy.deepcopy(baseline)
    for sparsity in args.sparsities:
        prune_unstructured(masked, sparsity)
        pruned = remove_masks(fine_tune(masked, loaders, args))
        record(f"unstructured_{round(sparsity * 100)}pct", "Unstructured pruning", pruned, pruned,
               sparsity)

    slim = baseline
    for keep in args.keep_fractions:
        widths = [max(1, round(width * keep)) for width in conv_widths(baseline)]
        slim = fine_tune(prune_structured(slim, widths), loaders, args)
        name = "structured_" + "-".join(map(str, widths))
        record(name, "Structured pruning", slim, slim)
        record(name + "_int8", "Structured pruning + INT8", quantize_int8(slim, calibration), slim)

    # Models are judged on validation data only; the test set never picks the winner.
    max_val_loss = max(args.max_val_loss, results[0]["val_loss"])
    for row in results:
        problems = []
        if row["val_accuracy"] < results[0]["val_accuracy"] - args.max_val_drop - 1e-9:
            problems.append(f"val acc {row['val_accuracy']:.1%}")
        if row["val_loss"] > max_val_loss:
            problems.append(f"val loss {row['val_loss']:.2f}")
        row["val_check"] = "fail: " + ", ".join(problems) if problems else "pass"
    passing = [row for row in results if row["val_check"] == "pass"]
    recommended = smallest(passing)
    best = [smallest(group) for technique in dict.fromkeys(r["technique"] for r in results)
            if (group := [r for r in passing if r["technique"] == technique])]

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "results.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "engine": engine, "torch": torch.__version__,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "recommended": recommended["model"], "results": results,
    }, indent=2), encoding="utf-8")
    plot_tradeoffs(output_dir / "compression_tradeoffs.png", results, best, recommended,
                   max_val_loss)

    print()
    print_table(results)
    print(f"\nRecommended: {recommended['model']} ({recommended['compression_ratio']:.1f}x smaller, "
          f"{recommended['test_accuracy']:.1%} test accuracy)")
    print(f"Saved results, figure, and models to: {output_dir}")


if __name__ == "__main__":
    main()
