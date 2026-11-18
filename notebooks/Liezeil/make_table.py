import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

FLASH_KB_BOARD = 2048
RAM_KB_BOARD = 786
TABLE_FILE = "final_table.md"


def load(name):
    return json.loads((HERE / "results" / name).read_text())


baseline = load("results_spec_baseline.json")
structured = load("results_structured.json")
unstructured = load("results_unstructured.json")
quantized = load("results_quantized.json")

rows = []


def add(name, accs, nonzero, flash, ram, ms):
    rows.append((name, np.mean(accs), np.std(accs), nonzero, flash, ram, ms))


add("Baseline (fp32)", baseline["test_accs"], 6099, 6099 * 4 / 1024, 183.5, quantized["baseline"]["ms_fp32"])
for name, label in [("unstructured_50", "Unstructured pruning 50%"),
                    ("unstructured_50_ft", "Unstructured 50% + fine-tune"),
                    ("unstructured_80_ft", "Unstructured 80% + fine-tune")]:
    r = unstructured[name]
    add(label, r["test_accs"], r["nonzero_params"], r["flash_kb"], r["ram_kb"], r["ms"])
for name, label in [("c1_4_c2_4", "Structured pruning 4/4"), ("c1_2_c2_4", "Structured pruning 2/4")]:
    r = structured[name]
    add(label, r["test_accs"], r["nonzero_params"], r["flash_kb"], r["ram_kb"], r["ms"])
r = quantized["baseline"]
add("Baseline + int8", r["int8_accs"], r["params"], r["flash_kb_int8"], r["ram_kb_int8"], r["ms_int8"])
for name, label in [("c1_4_c2_4", "Structured 4/4 + int8"), ("c1_2_c2_4", "Structured 2/4 + int8")]:
    r = quantized[name]
    add(label, r["int8_accs"], r["params"], r["flash_kb_int8"], r["ram_kb_int8"], r["ms_int8"])

base_flash = rows[0][4]
base_ram = rows[0][5]
lines = ["| Variant | Accuracy | Params (non-zero) | Flash (% of board) | RAM (% of board) | Flash smaller by | RAM smaller by | Speed (ms) |",
         "|---|---|---|---|---|---|---|---|"]
for name, mean, std, nonzero, flash, ram, ms in rows:
    lines.append(f"| {name} | {mean * 100:.1f}% +- {std * 100:.1f} | {nonzero:,} | "
                 f"{flash:.1f} KB ({flash / FLASH_KB_BOARD * 100:.1f}%) | {ram:.1f} KB ({ram / RAM_KB_BOARD * 100:.1f}%) | "
                 f"{base_flash / flash:.1f}x | {base_ram / ram:.1f}x | {ms:.2f} |")
lines += ["",
          "Accuracy: test set, mean over 5 seeds (+- spread in points). Smaller by: baseline divided by this row (1.0x = no saving).",
          "Structured a/b = filters in conv1 / conv2 (baseline is 8/16). Int8 sizes are estimates. Speed is laptop CPU, compare rows only."]
text = "\n".join(lines)
print(text)
(HERE / "results" / TABLE_FILE).write_text(text)
