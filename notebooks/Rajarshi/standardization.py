from scipy import io
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

train = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/train.mat'
test = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/test.mat'
val = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/val.mat'

train_data = io.loadmat(train)
test_data = io.loadmat(test)
val_data = io.loadmat(val)

# Filter out metadata
train_clean_data = {k: v for k, v in train_data.items() if not (k.startswith('__') and k.endswith('__'))}
# Filter out metadata
test_clean_data = {k: v for k, v in test_data.items() if not (k.startswith('__') and k.endswith('__'))}
val_clean_data = {k: v for k, v in val_data.items() if not (k.startswith('__') and k.endswith('__'))}

# Confirm the filter only dropped the metadata keys, not any actual values
def check_clean_matches_original(clean_data, original_data, labels_key):
    for key, value in clean_data.items():
        if key == labels_key:
            same = all(a[0] == b[0] for a, b in zip(value, original_data[key]))
        else:
            same = np.array_equal(value, original_data[key])
        print(f'{key}: {"match" if same else "MISMATCH"}')
# check_clean_matches_original(train_clean_data, train_data, 'trainLabels')

X_train = train_data['trainData']       # shape: (393, 5000)
X_test = test_data['testData']
X_val = val_data['valData']

y_train_npstr = [label[0] for label in train_data['trainLabels'].flatten()]
y_test_npstr = [label[0] for label in test_data['testLabels'].flatten()]
y_val_npstr = [label[0] for label in val_data['valLabels'].flatten()]

y_train = [i.astype(str).tolist() for i in y_train_npstr]
y_test = [i.astype(str).tolist() for i in y_test_npstr]
y_val = [i.astype(str).tolist() for i in y_val_npstr]

# ----------------------------------
# 
# Standardizing the Data
#
#----------------------------------

# We start with samlpe-wise z-score standardization
# Each sample will also have metadata, which would contain certain calculations that determine the properties of that sample
# The metadata can then be used as 'features' for the ML model
def standardize_samplewise(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / std

# Standardize the training set, validation set and test set
# since this is sample wise there is no data leakage
# We do this so that the model tries to find patterns in the actual features and differences of the sapmle and not the noise
X_train_std = standardize_samplewise(X_train)
X_val_std = standardize_samplewise(X_val)
X_test_std  = standardize_samplewise(X_test)

# Calculating metadata for each sample
# Shape of metadata would be an array of 7 values per sample
# This metadata acts as features for the sample and input features for the model
# [Mean, RMS, Peak Amplitude, Crest Factor, Standard Deviation, Kurtosis, Skewness]
X_train_metadata = np.zeros((X_train.shape[0], 7))
X_test_metadata = np.zeros((X_test.shape[0], 7))
X_val_metadata = np.zeros((X_val.shape[0], 7))

def metadata_extraction(dataset, meta_dataset):
    for idx, i in enumerate(dataset):
        mean = np.mean(i)
        RMS = np.sqrt(np.mean(i**2))
        peak = np.max(np.abs(i))
        crest = peak/RMS
        std = np.std(i)     # np.sqrt(sum((i - mean)**2) / len(i))
        kurtosis = (np.mean((i - mean)**4) / std**4) - 3    # (((1/len(i)) * sum( (i - mean)**4 )) / std**4) - 3
        skewness = np.mean((i - mean)**3) / std**3      # ((1/len(i)) * sum((i - mean)**3)) / std**3

        meta_dataset[idx] = [mean, RMS, peak, crest, std, kurtosis, skewness]

metadata_extraction(X_train, X_train_metadata)
metadata_extraction(X_test, X_test_metadata)
metadata_extraction(X_val, X_val_metadata)

# Standardizing the feature metadata for training, test and validation samples
# All the metadatas have varying scale, so we standardize them to put them on the same scale (mean = 0, std = 1)
# This makes sure no feature with a larger scale dominates the neural networks and all features are treated equally
train_mean = X_train_metadata.mean(axis=0)
train_std = X_train_metadata.std(axis=0)

# We standardize using the training data's mean and std as we don't want data to leak into the test and val
# If we standardized using the test and val data respectively, we would be assuming something about the data
# During inference, when we standardize a single sample, the standardization variables have to come from somewhere
# That somewhere can only be from a place we already know, which is only the training data
X_train_metadata_standard = (X_train_metadata - train_mean) / train_std
X_test_metadata_standard = (X_test_metadata - train_mean) / train_std
X_val_metadata_standard = (X_val_metadata - train_mean) / train_std


#-----------------------------------------
#
# Model Selection
#
#----------------------------------------

# 4. Initialize the SVM with RBF Kernel
# C: Controls the penalty for misclassification (lower = smoother boundary, more regularization)
# gamma: Controls the radius of influence for a single sample ('scale' is the default and a safe bet)

hyperparameters = []

svm_model = SVC(
    kernel='rbf', 
    C=1.0, 
    gamma='scale', 
    random_state=42
)

# 5. Train the model
svm_model.fit(X_train_metadata_standard, y_train)

# 6. Predict and Evaluate
y_pred = svm_model.predict(X_test_metadata_standard)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))

print("Original shape:", X_train.shape)
print("Standardized shape:", X_train_std.shape)
print("Mean of the first sample after scaling:", np.mean(X_train_std[0])) # Should be ~0
print("Std of the first sample after scaling:", np.std(X_train_std[0]))
print()

# Verify the standardization by checking the pooled distribution against a standard normal
flat = X_train_std.ravel()
print("Overall mean:", flat.mean(), "(expect ~0)")
print("Overall std:", flat.std(), "(expect ~1)")
print("Skewness:", stats.skew(flat), "(expect ~0 for a symmetric bell curve)")
print("Excess kurtosis:", stats.kurtosis(flat), "(expect ~0 for a normal-shaped tail)")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(flat, bins=100, density=True, color='#2a78d6', alpha=0.75, edgecolor='none', label='Standardized data')
x = np.linspace(flat.min(), flat.max(), 400)
ax.plot(x, stats.norm.pdf(x, 0, 1), color='#e34948', linewidth=1.5, label='Standard normal N(0,1)')
ax.set_title('Z-score Distribution (X_standardized)', color='#0b0b0b', fontsize=13, pad=12)
ax.set_xlabel('Z-score', color='#52514e')
ax.set_ylabel('Density', color='#52514e')
ax.legend(frameon=False, labelcolor='#52514e')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#c3c2b7')
ax.spines['bottom'].set_color('#c3c2b7')
ax.tick_params(colors='#52514e')
ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8)
ax.set_axisbelow(True)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')
fig.tight_layout()
plt.savefig('MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/standardized_zscore_histogram.png', dpi=150)
plt.show()

