import os

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR

SAMPLE_RATE = 48_828
class_order = ['InnerRaceFault', 'Normal', 'OuterRaceFault']

# load data
train = scipy.io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
val = scipy.io.loadmat(os.path.join(DATA_DIR, 'val.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_train = train['trainData']       # shape: (393, 5000)
y_train = [label[0] for label in train['trainLabels'].flatten()]
X_val = val['valData']
X_test = test['testData']

time = np.arange(X_train.shape[1]) / SAMPLE_RATE


# global standrdization, preserves amplitude diffs between signals
train_mean = np.mean(X_train)
train_std = np.std(X_train)

X_train_global = (X_train - train_mean) / train_std # np
X_val_global = (X_val - train_mean) / train_std
X_test_global = (X_test - train_mean) / train_std

print('Global standardization')
print('  train mean/std used:', train_mean, train_std)
print('  scaled train mean/std:', np.mean(X_train_global), np.std(X_train_global))
print('  scaled val mean/std:', np.mean(X_val_global), np.std(X_val_global))
print('  scaled test mean/std:', np.mean(X_test_global), np.std(X_test_global))


# per sample standardization, each window normalized, removes apltiude differences between signals, keeps
# internal shape
def standardize_samplewise(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / std

X_train_persample = standardize_samplewise(X_train)
X_val_persample = standardize_samplewise(X_val)
X_test_persample = standardize_samplewise(X_test)

print('\nPer-sample standardization')
print('  mean/std of sample 0:', np.mean(X_train_persample[0]), np.std(X_train_persample[0]))
print('  mean/std of sample 1:', np.mean(X_train_persample[1]), np.std(X_train_persample[1]))
