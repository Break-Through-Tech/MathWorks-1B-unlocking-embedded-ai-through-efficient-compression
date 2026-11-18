# Two-panel summary: accuracy vs remaining parameters, and weight memory per technique
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from pathlib import Path

HERE = Path(__file__).resolve().parent
load = lambda n: json.loads((HERE / "results" / n).read_text())
base, st, un, q = (load("results_spec_baseline.json"), load("results_structured.json"),
                   load("results_unstructured.json"), load("results_quantized.json"))
acc = lambda a: float(np.mean(a)) * 100

BASE_PARAMS, BASE_KB = 6099, 6099 * 4 / 1024
SURFACE, TEXT, LIGHT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c2c1b4"
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
