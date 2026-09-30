import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
data = np.load(HERE / "standardized_global.npz")

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

# One SVM
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

print("val accuracy: ", model.score(X_val, y_val))
print("test accuracy:", model.score(X_test, y_test))

# Compare kernels
for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    print(f"{kernel:8} val {model.score(X_val, y_val):.3f}  test {model.score(X_test, y_test):.3f}")

# Using frequency domain
def to_spectrum(X):
    return np.abs(np.fft.rfft(X, axis=1))   # (N, 5000) -> (N, 2501)

S_train = to_spectrum(X_train)
S_val = to_spectrum(X_val)
S_test = to_spectrum(X_test)

# Standardize each frequency using TRAIN statistics only
scaler = StandardScaler().fit(S_train)
S_train = scaler.transform(S_train)
S_val = scaler.transform(S_val)
S_test = scaler.transform(S_test)

for kernel in ["linear", "rbf"]:
    model = SVC(kernel=kernel)
    model.fit(S_train, y_train)
    print(f"FFT {kernel:6} val {model.score(S_val, y_val):.3f}  test {model.score(S_test, y_test):.3f}")

model = SVC(kernel="linear").fit(S_train, y_train)
print("linear coef shape:", model.coef_.shape)
print("linear support vectors:", model.support_vectors_.shape)

model = SVC(kernel="rbf").fit(S_train, y_train)
print("rbf support vectors:", model.support_vectors_.shape)