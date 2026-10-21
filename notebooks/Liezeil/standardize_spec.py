# Turns the signals into standardized spectrograms and saves them
from pathlib import Path
import numpy as np
from spectro import make_all

DATA_FILE = Path(__file__).parent / "standardized_global.npz"
N_FFT = 128
HOP = 64
OUT_FILE = Path(__file__).parent / "spectrograms_128.npz"

data = np.load(DATA_FILE)
spec_train = make_all(data["X_train"], N_FFT, HOP)

print(spec_train.shape)

mean = spec_train.mean()
std = spec_train.std()

print(mean, std)

spec_val = make_all(data["X_val"], N_FFT, HOP)
spec_test = make_all(data["X_test"], N_FFT, HOP)

def standardize(spec, mean, std):
    return (spec - mean) / std

for name, spec in [("train", spec_train), ("val", spec_val), ("test", spec_test)]:
    z = standardize(spec, mean, std)
    print(name, z.shape, z.mean().round(3), z.std().round(3), z.max().round(1))

z = standardize(spec_train, mean, std)
for c, name in enumerate(["Normal", "InnerRaceFault", "OuterRaceFault"]):
    zc = z[data["y_train"] == c]
    print(name, zc.max().round(1), (zc > 5).mean().round(4))

np.savez(
    OUT_FILE,
    X_train=standardize(spec_train, mean, std).astype(np.float32),
    X_val=standardize(spec_val, mean, std).astype(np.float32),
    X_test=standardize(spec_test, mean, std).astype(np.float32),
    y_train=data["y_train"], y_val=data["y_val"], y_test=data["y_test"],
    mean=mean, std=std,
)
print("saved", OUT_FILE.name)