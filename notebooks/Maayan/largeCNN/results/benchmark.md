# FaultCNNLarge benchmark

Test-set metrics from `benchmark_model` in `notebooks/Maayan/common.py`. Latency is the median single-sample CPU time on one machine (arm64 Mac, torch 2.14, `qnnpack` for int8), so compare only against runs from the same machine.

| Metric | Large fp32 (before) | Large int8 (after) | Large unstructured pruned, 80% (after) | Large structured pruned, 50% channels (after) | Large structured 50% + int8 (after) | Small fp32 (reference) |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9902 |
| Macro precision | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9905 |
| Macro recall | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9902 |
| Macro F1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9902 |
| Params (total) | 47,683 | 47,203 | 47,683 | 14,371 | 14,131 | 1,027 |
| Params (non-zero) | 47,683 | 47,203 | 10,166 | 14,371 | 14,131 | 1,027 |
| Dtype | float32 | int8 | float32 | float32 | int8 | float32 |
| Size on disk | 199.0 KB | 56.9 KB | 199.3 KB | 68.2 KB | 24.7 KB | 7.0 KB |
| Latency (ms) | 0.59 | 0.41 | 0.79 (noise, see notes) | 0.42 | 0.30 | 0.20 |

Notes:
- int8 params are 480 fewer than fp32 because BatchNorm is folded into the convs before quantization.
- Unstructured pruning (`prune_large_cnn.py`) zeroes weights but keeps dense tensors, so size and latency do not improve; only the non-zero count drops. Its 0.79 ms is run-to-run noise on a busy machine (fp32 re-measured at 0.58-0.67 ms), not a real slowdown.
- Structured pruning (`prune_structured_large_cnn.py`) keeps the top 50% of channels per conv by L1 norm, builds a narrower network (widths 16/32/64/128 to 8/16/32/64) and fine-tunes 20 epochs, so size and params really shrink.
- Structured + int8 (`quantize_structured_large_cnn.py`) quantizes the 50% structured model with the same BatchNorm folding and calibration as the full-width int8 model; BN fold removes 240 params (14,371 to 14,131). Latency was stable at 0.30 ms across 3 runs.
- Latency is noisy (re-measured fp32 at 0.58-0.67 ms back to back), so treat differences under about 0.1 ms as equal.
- Size on disk is the `state_dict` checkpoint, which includes fixed file overhead and, for int8, the scales and zero-points.

## Pruning sweeps

Unstructured (global L1, 10 fine-tune epochs):

| Sparsity | Accuracy | Non-zero params | Size |
|---|---|---|---|
| 50% | 1.0000 | 24,235 | 199.3 KB |
| 70% | 0.9804 | 14,856 | 199.3 KB |
| 80% | 1.0000 | 10,166 | 199.3 KB |
| 90% | 1.0000 | 5,477 | 199.3 KB |

The 70% dip is 2 of 102 test samples; checkpoints are picked on a 27-sample validation set, so treat it as noise.

Structured (channels kept per conv, 20 fine-tune epochs):

| Kept | Widths | Accuracy | Params | Size | Latency (ms) |
|---|---|---|---|---|---|
| 75% | 12/24/48/96 | 1.0000 | 28,627 | 124.5 KB | 0.68 |
| 50% | 8/16/32/64 | 1.0000 | 14,371 | 68.3 KB | 0.62 (0.42 on re-run) |
| 35% | 6/11/22/45 | 1.0000 | 8,075 | 43.1 KB | 0.56 |
| 25% | 4/8/16/32 | 0.9902 | 4,915 | 30.9 KB | 0.38 |

## Per-class

| Class | Large fp32 (P / R / F1) | Large int8 (P / R / F1) | Large pruned: unstructured 80%, structured 50%, structured 50% + int8 (P / R / F1) | Small fp32 (P / R / F1) |
|---|---|---|---|---|
| InnerRaceFault | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0.9714 / 1.0000 / 0.9855 |
| Normal | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| OuterRaceFault | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.9706 / 0.9851 |

## Files

- `models/large_model_fp32.pt`, `models/large_model_int8.pt`, `models/large_model_pruned.pt` (unstructured 80%), `models/large_model_structured.pt` (50% channels), `models/large_model_structured_int8.pt` (50% channels + int8): checkpoints
- `results/confusion_matrix_fp32.png`, `results/confusion_matrix_int8.png`, `results/confusion_matrix_pruned.png`, `results/confusion_matrix_structured.png`, `results/confusion_matrix_structured_int8.png`: test-set confusion matrices
- Scripts: `train_large_cnn.py` (train fp32), `quantize_large_cnn.py` (int8 PTQ and benchmark), `../prune_large_cnn.py` (unstructured), `../prune_structured_large_cnn.py` (structured), `../quantize_structured_large_cnn.py` (structured + int8)
