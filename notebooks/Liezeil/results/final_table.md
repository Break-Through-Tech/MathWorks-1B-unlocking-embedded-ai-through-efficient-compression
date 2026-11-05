| Variant | Accuracy | Params (non-zero) | Flash (% of board) | RAM (% of board) | Flash smaller by | RAM smaller by | Speed (ms) |
|---|---|---|---|---|---|---|---|
| Baseline (fp32) | 100.0% +- 0.0 | 6,099 | 23.8 KB (1.2%) | 183.5 KB (23.3%) | 1.0x | 1.0x | 0.30 |
| Unstructured pruning 50% | 87.3% +- 10.9 | 3,135 | 23.8 KB (1.2%) | 183.5 KB (23.3%) | 1.0x | 1.0x | 0.26 |
| Unstructured 50% + fine-tune | 100.0% +- 0.0 | 3,135 | 23.8 KB (1.2%) | 183.5 KB (23.3%) | 1.0x | 1.0x | 0.23 |
| Unstructured 80% + fine-tune | 100.0% +- 0.0 | 1,357 | 23.8 KB (1.2%) | 183.5 KB (23.3%) | 1.0x | 1.0x | 0.25 |
| Structured pruning 4/4 | 100.0% +- 0.0 | 1,551 | 6.1 KB (0.3%) | 93.4 KB (11.9%) | 3.9x | 2.0x | 0.12 |
| Structured pruning 2/4 | 99.6% +- 0.5 | 1,455 | 5.7 KB (0.3%) | 56.5 KB (7.2%) | 4.2x | 3.2x | 0.09 |
| Baseline + int8 | 100.0% +- 0.0 | 6,099 | 6.0 KB (0.3%) | 45.9 KB (5.8%) | 4.0x | 4.0x | 0.17 |
| Structured 4/4 + int8 | 100.0% +- 0.0 | 1,551 | 1.5 KB (0.1%) | 23.3 KB (3.0%) | 15.7x | 7.9x | 0.14 |
| Structured 2/4 + int8 | 99.6% +- 0.5 | 1,455 | 1.4 KB (0.1%) | 14.1 KB (1.8%) | 16.8x | 13.0x | 0.14 |

Accuracy: test set, mean over 5 seeds (+- spread in points). Smaller by: baseline divided by this row (1.0x = no saving).
Structured a/b = filters in conv1 / conv2 (baseline is 8/16). Int8 sizes are estimates. Speed is laptop CPU, compare rows only.