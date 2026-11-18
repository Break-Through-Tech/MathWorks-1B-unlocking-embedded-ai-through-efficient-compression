import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
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



SURFACE, TEXT, LIGHT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c2c1b4"
TEXT_LIGHT = LIGHT

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT_LIGHT, "xtick.color": TEXT_LIGHT, "ytick.color": TEXT,
    "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID, "axes.grid": False,
})

names = [f"{r[0]}\n{r[1] * 100:.1f}% accuracy" for r in rows]
flash = [r[4] for r in rows]
ram = [r[5] for r in rows]
positions = list(range(len(rows)))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
for ax, values, title, unit in [(axes[0], flash, "Flash (weights)", "KB"), (axes[1], ram, "RAM (activations)", "KB")]:
    ax.barh(positions, values, color=BLUE, height=0.6)
    for y, v in zip(positions, values):
        ax.text(v + max(values) * 0.015, y, f"{v:.1f}", va="center", color=TEXT, fontsize=12)
    ax.set_title(f"{title}, {unit}", loc="left", color=TEXT, fontsize=15)
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
axes[0].set_yticks(positions)
axes[0].set_yticklabels(names)
fig.suptitle("Structured pruning and int8 shrink the model; unstructured pruning does not", x=0.01, ha="left", fontsize=17)
plt.tight_layout()
plt.savefig(HERE / "figures" / "slide_size.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(11, 6))
for name, mean, std, nonzero, f, r, ms in rows:
    if name.startswith("Unstructured"):
        continue
    color = ORANGE if "int8" in name else BLUE
    ax.errorbar(r, mean * 100, yerr=std * 100, fmt="o", color=color, markersize=11,
                markeredgecolor=SURFACE, markeredgewidth=2, capsize=4, linewidth=2)
    below = name in ("Baseline (fp32)", "Baseline + int8")
    ax.annotate(name, (r, mean * 100), textcoords="offset points", xytext=(6, -28) if below else (10, 12),
                ha="right" if name == "Baseline + int8" else ("center" if below else "left"), fontsize=12, color=TEXT)
ax.set_xscale("log")
ax.set_xticks([15, 25, 50, 100, 180])
ax.set_xticklabels(["15", "25", "50", "100", "180"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlim(10, 330)
ax.set_ylim(98.9, 100.3)
ax.set_xlabel("RAM needed (KB, log scale; lower is better)")
ax.set_ylabel("Test accuracy (%)")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.plot([], [], "o", color=BLUE, label="float32")
ax.plot([], [], "o", color=ORANGE, label="int8")
ax.legend(frameon=False, loc="lower left")
ax.set_title("Less RAM costs almost no accuracy (y-axis starts at 98.9%)", loc="left", fontsize=16)
plt.tight_layout()
plt.savefig(HERE / "figures" / "slide_accuracy_vs_ram.png", dpi=200)
plt.close()


base, st, un, q = baseline, structured, unstructured, quantized
acc = lambda a: float(np.mean(a)) * 100
BASE_PARAMS, BASE_KB = 6099, 6099 * 4 / 1024
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": TEXT, "axes.labelcolor": LIGHT, "xtick.color": LIGHT, "ytick.color": TEXT,
                     "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID})

# Left: accuracy as parameters are removed
s_pts = sorted([(BASE_PARAMS, acc(base["test_accs"]))] + [(r["nonzero_params"], acc(r["test_accs"])) for r in st.values()], reverse=True)
u_pts = sorted([(BASE_PARAMS, acc(base["test_accs"]))] + [(un[k]["nonzero_params"], acc(un[k]["test_accs"])) for k in ("unstructured_50_ft", "unstructured_80_ft")], reverse=True)

# Right: weight memory (KB) for the smallest 100%-accuracy model of each technique
bars = [("Baseline (FP32)\n8-16-32 channels", BASE_KB, acc(base["test_accs"]), GREY),
        ("INT8 quantization\n8-16-32 channels", q["baseline"]["flash_kb_int8"], acc(q["baseline"]["int8_accs"]), GREY),
        ("Unstructured pruning\n80% of weights zeroed", un["unstructured_80_ft"]["flash_kb"], acc(un["unstructured_80_ft"]["test_accs"]), GREY),
        ("Structured pruning\n4-4-32 channels", st["c1_4_c2_4"]["flash_kb"], acc(st["c1_4_c2_4"]["test_accs"]), GREY),
        ("Structured pruning + INT8\n4-4-32 channels  ·  recommended", q["c1_4_c2_4"]["flash_kb_int8"], acc(q["c1_4_c2_4"]["int8_accs"]), BLUE)]

fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.25]})
for pts, c, m, lab in [(s_pts, BLUE, "o", "Structured pruning (FP32)"), (u_pts, ORANGE, "s", "Unstructured pruning (FP32)")]:
    a.plot(*zip(*pts), color=c, marker=m, markersize=8, linewidth=2, label=lab)
a.set_xscale("log"); a.invert_xaxis()
ticks = [6000, 3000, 1500]
a.xaxis.set_major_locator(FixedLocator(ticks)); a.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in ticks]))
a.xaxis.set_minor_formatter(FixedFormatter([]))
a.set_xlabel(f"Nonzero parameters, log scale (baseline {BASE_PARAMS:,})"); a.set_ylabel("Test accuracy (%)")
a.set_ylim(98, 100.5); a.yaxis.grid(True, color=GRID); a.set_axisbelow(True); a.tick_params(length=0)
a.legend(frameon=False, loc="lower left")
a.set_title("Test accuracy as pruning removes parameters", loc="left", fontsize=14)

ys = list(range(len(bars)))[::-1]
for y, (name, kb, ac, col) in zip(ys, bars):
    b.barh(y, kb, color=col, height=0.5)
    b.text(kb + 0.4, y, f"{kb:.1f} KB  ·  {BASE_KB / kb:.1f}x smaller  ·  {ac:.0f}% test acc.", va="center", fontsize=11)
b.set_yticks(ys); b.set_yticklabels([x[0] for x in bars]); b.set_xlim(0, BASE_KB * 2.3)
b.set_xlabel("Weight memory (KB)"); b.xaxis.grid(True, color=GRID); b.set_axisbelow(True); b.tick_params(length=0)
b.set_title("Weight memory: smallest model per technique", loc="left", fontsize=14)
fig.text(0.01, 0.01, "Test accuracy on 102 held-out signals, mean over 5 seeds. Weight memory = parameters x bytes (INT8 sizes are estimates). "
         "Unstructured pruning keeps zeros in storage, so memory is unchanged.", fontsize=9, color=LIGHT)
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(HERE / "figures" / "pruning_summary.png", dpi=200)
