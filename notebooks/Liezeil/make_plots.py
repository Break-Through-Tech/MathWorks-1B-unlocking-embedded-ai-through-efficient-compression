import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pathlib import Path
from make_table import rows, FLASH_KB_BOARD, RAM_KB_BOARD

HERE = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LIGHT = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

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
