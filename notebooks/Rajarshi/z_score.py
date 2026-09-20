from scipy import io
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/train.mat'
test = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/test.mat'
val = 'MathWorks-1B-unlocking-embedded-ai-through-efficient-compression/data/val.mat'

train_data = io.loadmat(train)
test_data = io.loadmat(test)
val_data = io.loadmat(val)

train_data.keys()

data = np.array([])
for i in train_data['trainData'][0]:
    data = np.append(data, i)

y = np.linspace(0, data.size, num=5000)
z_scores = stats.zscore(data)
plt.plot(y, data)
plt.show()

sns.histplot(z_scores, kde=True, stat="density", color="#4C72B0", alpha=0.6, bins=30)
sns.histplot(z_scores, kde=True, stat="density")
plt.show()

