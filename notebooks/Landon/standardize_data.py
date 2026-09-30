"""Standardize vibration signals using training-only statistics.

Run: python standardize_data.py
Outputs: data/standardized/{train,val,test}.mat and standardization.mat.
Original data files are preserved. Running again replaces the generated files.
Requires NumPy and SciPy (included in the mathworks1b environment).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = DATA_DIR / "standardized"


def main():
    datasets = {}
    for split in ("train", "val", "test"):
        contents = loadmat(DATA_DIR / f"{split}.mat")
        signals = np.asarray(contents[f"{split}Data"], dtype=np.float64)
        labels = contents[f"{split}Labels"]
        if signals.ndim != 2 or signals.size == 0:
            raise ValueError(f"{split}Data must be a nonempty 2D signal matrix.")
        if not np.isfinite(signals).all():
            raise ValueError(f"{split}Data contains NaN or infinite values.")
        if labels.size != signals.shape[0]:
            raise ValueError(f"{split} labels do not match the number of signals.")
        datasets[split] = (signals, labels)

    training_signals = datasets["train"][0]

    # These are single-channel signals: pool all training segments and time
    # points to get one mean and standard deviation. Scaling every signal by
    # the same values preserves relative amplitudes between segments/classes.
    training_mean = float(training_signals.mean())
    training_std = float(training_signals.std(ddof=0))
    if not np.isfinite(training_std) or training_std <= 0:
        raise ValueError("Training standard deviation must be finite and positive.")
    if not np.isfinite(training_mean):
        raise ValueError("Training mean must be finite.")
    for split, (signals, _) in datasets.items():
        if signals.shape[1] != training_signals.shape[1]:
            raise ValueError(f"{split} signal length differs from training data.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save the fitted values so new signals can use the exact same transform:
    # standardized_signal = (signal - training_mean) / training_std
    savemat(OUTPUT_DIR / "standardization.mat", {
        "training_mean": training_mean,
        "training_std": training_std,
        "ddof": 0,
        "method": "global training-only z-score",
    })

    print(f"Training mean: {training_mean:.8g}")
    print(f"Training standard deviation: {training_std:.8g}")
    for split, (signals, labels) in datasets.items():
        standardized = (signals - training_mean) / training_std
        savemat(OUTPUT_DIR / f"{split}.mat", {
            f"{split}Data": standardized,
            f"{split}Labels": labels,
        }, do_compression=True)
        print(f"{split}: shape={standardized.shape}, "
              f"mean={standardized.mean():.6f}, "
              f"std={standardized.std(ddof=0):.6f}")

    # Only training data is expected to have mean 0 and standard deviation 1.
    print(f"Saved standardized data and fitted statistics to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
