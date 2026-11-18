| Method | Accuracy | Storage | RAM |
|---|---|---|---|
| Baseline | 100% | 23.8 KB | 183 KB |
| Zero out 80% of weights, then fine-tune | 100% | 23.8 KB (no change) | 183 KB (no change) |
| Remove filters (4/4) | 100% | 6.1 KB | 93 KB |
| Remove filters + INT8 | **100%** | **1.5 KB** | **23 KB** |

Accuracy is test accuracy, averaged over 5 seeds. INT8 sizes are estimates. Source: `results/final_table.md`.
