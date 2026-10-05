import os

import scipy.io
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = SCRIPT_DIR

class_order = ['InnerRaceFault', 'Normal', 'OuterRaceFault']

# ---- Hyperparameters (tune these) ----
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 5
CRITERION = 'gini'
RANDOM_STATE = 42

# ---- Load data ----
train = scipy.io.loadmat(os.path.join(DATA_DIR, 'train.mat'))
val = scipy.io.loadmat(os.path.join(DATA_DIR, 'val.mat'))
test = scipy.io.loadmat(os.path.join(DATA_DIR, 'test.mat'))

X_train = train['trainData']       # shape: (393, 5000), used as-is - no feature extraction
y_train = [label[0] for label in train['trainLabels'].flatten()]
X_val = val['valData']
y_val = [label[0] for label in val['valLabels'].flatten()]
X_test = test['testData']
y_test = [label[0] for label in test['testLabels'].flatten()]

# ---- Train decision tree directly on the 5,000 raw columns ----
clf = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    criterion=CRITERION,
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)

# ---- Evaluate ----
for name, X, y in [('Validation', X_val, y_val), ('Test', X_test, y_test)]:
    y_pred = clf.predict(X)
    print(f'--- {name} ---')
    print(f'Accuracy: {accuracy_score(y, y_pred):.4f}')
    print(classification_report(y, y_pred, labels=class_order, zero_division=0))
    print()

# ---- Confusion matrix on test set ----
y_test_pred = clf.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, labels=class_order, ax=ax)
ax.set_title('Decision Tree Confusion Matrix - Raw Signal (Test)')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '09_decision_tree_raw_confusion_matrix.png'), dpi=150)
plt.close(fig)

print('Tree depth:', clf.get_depth())
print('Number of leaves:', clf.get_n_leaves())
print('Saved confusion matrix to', OUT_DIR)
