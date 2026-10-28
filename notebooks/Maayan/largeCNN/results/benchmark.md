# FaultCNNLarge benchmark

Test-set metrics from `benchmark_model` in `notebooks/Maayan/common.py`. Latency is the median single-sample CPU time on one machine (arm64 Mac, torch 2.14, `qnnpack` for int8), so compare only against runs from the same machine.

| Metric | Large fp32 (before) | Large int8 (after) | Large pruned (after) | Small fp32 (reference) |
|---|---|---|---|---|
| Accuracy | 1.0000 | 1.0000 | TBD | 0.9902 |
| Macro precision | 1.0000 | 1.0000 | TBD | 0.9905 |
| Macro recall | 1.0000 | 1.0000 | TBD | 0.9902 |
| Macro F1 | 1.0000 | 1.0000 | TBD | 0.9902 |
| Params (total) | 47,683 | 47,203 | TBD | 1,027 |
| Params (non-zero) | 47,683 | 47,203 | TBD | 1,027 |
| Dtype | float32 | int8 | TBD | float32 |
| Size on disk | 199.0 KB | 56.9 KB | TBD | 7.0 KB |
| Latency (ms) | 0.59 | 0.41 | TBD | 0.20 |

Notes:
- int8 params are 480 fewer than fp32 because BatchNorm is folded into the convs before quantization.
- Size on disk is the `state_dict` checkpoint, which includes fixed file overhead and, for int8, the scales and zero-points.

## Per-class

| Class | Large fp32 (P / R / F1) | Large int8 (P / R / F1) | Small fp32 (P / R / F1) |
|---|---|---|---|
| InnerRaceFault | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 0.9714 / 1.0000 / 0.9855 |
| Normal | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| OuterRaceFault | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.9706 / 0.9851 |

## Files

- `models/large_model_fp32.pt`, `models/large_model_int8.pt`: checkpoints
- `results/confusion_matrix_fp32.png`, `results/confusion_matrix_int8.png`: test-set confusion matrices
- Scripts: `train_large_cnn.py` (train fp32), `quantize_large_cnn.py` (int8 PTQ and benchmark)
