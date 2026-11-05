import os

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR

class_order = ['InnerRaceFault', 'Normal', 'OuterRaceFault']

# hyperparams
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 5
CRITERION = 'gini'
RANDOM_STATE = 42

# load
train = scipy.io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
val = scipy.io.loadmat(os.path.join(DATA_DIR, 'val.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_train = train['trainData']
y_train = [label[0] for label in train['trainLabels'].flatten()]
X_val = val['valData']
y_val = [label[0] for label in val['valLabels'].flatten()]
X_test = test['testData']
y_test = [label[0] for label in test['testLabels'].flatten()]

# current features: rms, kurtosis, peak to peak, and sd
def extract_features(X):
    return pd.DataFrame({
        'rms': np.sqrt(np.mean(X ** 2, axis=1)),
        'kurtosis': kurtosis(X, axis=1),
        'peak_to_peak': np.ptp(X, axis=1),
        'sd': np.std(X, axis=1),
        'skewness': skew(X, axis=1)
    })

X_train_feat = extract_features(X_train)
X_val_feat = extract_features(X_val)
X_test_feat = extract_features(X_test)

# dt
clf = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    criterion=CRITERION,
    random_state=RANDOM_STATE,
)
clf.fit(X_train_feat, y_train)

#  Evaluate 
for name, X, y in [('Validation', X_val_feat, y_val), ('Test', X_test_feat, y_test)]:
    y_pred = clf.predict(X)
    print(f'--- {name} ---')
    print(f'Accuracy: {accuracy_score(y, y_pred):.4f}')
    print(classification_report(y, y_pred, labels=class_order))
    print()

# feature importances
importances = pd.Series(clf.feature_importances_, index=X_train_feat.columns).sort_values(ascending=False)
print('Feature importances:') # kurtosis + rms
print(importances)
print()

# confusion matrix
y_test_pred = clf.predict(X_test_feat)
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, labels=class_order, ax=ax)
ax.set_title('Decision Tree Confusion Matrix (Test)')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '07_decision_tree_confusion_matrix.png'), dpi=150)
plt.close(fig)

print('Tree depth:', clf.get_depth())
print('Number of leaves:', clf.get_n_leaves())
print('Saved confusion matrix to', OUT_DIR)
