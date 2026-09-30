""" Standardize the bearing data and save it for training"""
from pathlib import Path
import numpy as np
import scipy.io


HERE = Path(__file__).resolve().parent # Folder this script lives in
DATA_DIR = HERE.parent.parent / "data" # Repo's data folder

CLASS_NAMES = ["Normal", "InnerRaceFault", "OuterRaceFault"]

def load_split(name):
    """Load 'train', 'val' or 'test'. Returns the signals and number labels."""
    file = scipy.io.loadmat(DATA_DIR / f"{name}.mat")
    signals = file[f"{name}Data"] # shape: (samples, 5000)

    # label[0] pulls out the text since MATLAB wraps each label in an array
    label_names = [label[0] for label in file[f"{name}Labels"].flatten()]
    labels = np.array([CLASS_NAMES.index(n) for n in label_names])
    return signals, labels

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

# Standardize: (value - mean) / std
def standardize(X):
    return (X - X_train.mean()) / X_train.std()

X_train_std = standardize(X_train)
X_val_std = standardize(X_val)
X_test_std = standardize(X_test)

# Check: train should be ~0 mean and 1 std, val/test should be close
for name, before, after in [("train", X_train, X_train_std),
                            ("val", X_val, X_val_std),
                            ("test", X_test, X_test_std)]:
    print(f"{name:5} before: mean {before.mean():.3f}, std {before.std():.3f} | "
          f"after: mean {after.mean():.3f}, std {after.std():.3f}")


# Check: each class should show up about equally often in the train labels
print("train label counts:", np.bincount(y_train))

# Save everything for train.py
np.savez(HERE / "standardized_global.npz",
         X_train=X_train_std, y_train=y_train,
         X_val=X_val_std, y_val=y_val,
         X_test=X_test_std, y_test=y_test)
print("saved standardized_global.npz")