import scipy.io as sio
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(filename, data_key, label_key):
    """Load one .mat split and unwrap MATLAB cell-array labels into plain strings."""
    mat = sio.loadmat(DATA_DIR / filename)
    X = mat[data_key]
    y_raw = mat[label_key]
    y = np.array([item[0] if isinstance(item, np.ndarray) else item for item in y_raw.ravel()])
    return X, y


def standardize(X_train, X_val, X_test):
    """Global standardization, fit on train only."""
    mean = X_train.mean()
    std = X_train.std()
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def main():
    X_train, y_train = load_split("train.mat", "trainData", "trainLabels")
    X_val, y_val = load_split("val.mat", "valData", "valLabels")
    X_test, y_test = load_split("test.mat", "testData", "testLabels")

    X_train_std, X_val_std, X_test_std, mean, std = standardize(X_train, X_val, X_test)

    np.save(OUT_DIR / "X_train.npy", X_train_std)
    np.save(OUT_DIR / "X_val.npy", X_val_std)
    np.save(OUT_DIR / "X_test.npy", X_test_std)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "y_test.npy", y_test)

    # save the fit parameters too, for reproducibility / inference later
    np.save(OUT_DIR / "standardization_params.npy", {"mean": mean, "std": std})

    print(f"Saved standardized arrays to {OUT_DIR}/")
    print(f"Train mean/std used: {mean:.4f} / {std:.4f}")
    print(f"Shapes: train {X_train_std.shape}, val {X_val_std.shape}, test {X_test_std.shape}")


if __name__ == "__main__":
    main()