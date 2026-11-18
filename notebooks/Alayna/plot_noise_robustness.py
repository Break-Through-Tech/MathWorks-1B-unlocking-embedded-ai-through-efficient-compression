from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd() / "robustness_outputs"
summary = pd.read_csv(OUTPUT_DIR / "noise_robustness_summary.csv", index_col="noise_level")
level_order = ["Clean", "30 dB", "20 dB", "10 dB", "5 dB", "0 dB", "-5 dB"]
summary = summary.reindex(level_order)
x = np.arange(len(level_order))

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(9, 4.8))
acc_pct = summary["accuracy_mean"].to_numpy() * 100
sd_pct = summary["accuracy_sd"].to_numpy() * 100
ax.plot(x, acc_pct, marker="o", lw=2.5, ms=6)
ax.fill_between(x, np.maximum(0, acc_pct - sd_pct), np.minimum(100, acc_pct + sd_pct), alpha=0.17)
ax.set_xticks(x, level_order)
ax.set_ylabel("Test accuracy (%)")
ax.set_xlabel("Added Gaussian noise (lower SNR = stronger noise)")
ax.set_title("Does bearing-fault accuracy survive noisier sensor readings?", loc="left", weight="bold", pad=14)
ax.set_ylim(max(0, np.floor((np.min(acc_pct - sd_pct) - 5) / 10) * 10), 101)
ax.grid(axis="y", alpha=0.22)
fig.text(0.01, -0.005, "102 original test signals · 10 random noise draws per SNR · shaded band = ±1 SD", color="dimgray", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "accuracy_vs_noise.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.8))
for key, name in [
    ("normal_recall", "Normal"),
    ("inner_recall", "Inner race fault"),
    ("outer_recall", "Outer race fault"),
]:
    ax.plot(x, summary[key].to_numpy() * 100, marker="o", lw=2.25, ms=5, label=name)
ax.set_xticks(x, level_order)
ax.set_ylabel("Class recall (%)")
ax.set_xlabel("Added Gaussian noise (lower SNR = stronger noise)")
ax.set_title("Which bearing conditions become harder to recognize?", loc="left", weight="bold", pad=14)
ax.set_ylim(max(0, np.floor((min(summary[["normal_recall", "inner_recall", "outer_recall"]].min()) * 100 - 7) / 10) * 10), 101)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, loc="lower left")
fig.text(0.01, -0.005, "Mean recall over 10 random noise draws per SNR (clean = 1 draw)", color="dimgray", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "recall_by_class_vs_noise.png", bbox_inches="tight")
plt.close(fig)

print("Saved:", OUTPUT_DIR / "accuracy_vs_noise.png")
print("Saved:", OUTPUT_DIR / "recall_by_class_vs_noise.png")
