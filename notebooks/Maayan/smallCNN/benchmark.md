# FaultCNN (small) benchmark

Test-set metrics from `benchmark_model` in `notebooks/Maayan/common.py`. Latency is the median single-sample CPU time on one machine, so compare only against runs from the same machine.

| Metric | Value |
|---|---|
| Accuracy | 0.9902 |
| Macro precision | 0.9905 |
| Macro recall | 0.9902 |
| Macro F1 | 0.9902 |
| Params (total) | 1,027 |
| Params (non-zero) | 1,027 |
| Dtype | float32 |
| Size on disk | 7.0 KB |
| Latency (ms) | 0.20 |

## Per-class

| Class | Precision | Recall | F1 |
|---|---|---|---|
| InnerRaceFault | 0.9714 | 1.0000 | 0.9855 |
| Normal | 1.0000 | 1.0000 | 1.0000 |
| OuterRaceFault | 1.0000 | 0.9706 | 0.9851 |
