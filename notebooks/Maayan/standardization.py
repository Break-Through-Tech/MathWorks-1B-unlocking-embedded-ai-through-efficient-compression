import os
import numpy as np
from scipy import io

DATA_DIR = 'data'
OUT_DIR = 'data'


def load_split(name):
    """Load the data matrix and unwrapped string labels for a split ('train'/'val'/'test')."""
    mat = io.loadmat(os.path.join(DATA_DIR, f'{name}.mat'))
    data = mat[f'{name}Data']
    labels = np.array([label[0] for label in mat[f'{name}Labels'].flatten()])
    return data, labels


def standardize(train_data, val_data, test_data):
    """Z-score every split using a single global mean/std fit on train_data only.

    Center train_data at mean = 0 and standard deviation = 1
    """
    mu = train_data.mean()
    sigma = train_data.std()
    scale = lambda data: (data - mu) / sigma
    return scale(train_data), scale(val_data), scale(test_data), mu, sigma


if __name__ == '__main__':
    train_data, train_labels = load_split('train')
    val_data, val_labels = load_split('val')
    test_data, test_labels = load_split('test')

    train_scaled, val_scaled, test_scaled, mu, sigma = standardize(train_data, val_data, test_data)

    print(f'Train stats used for scaling: mean={mu:.4f}, std={sigma:.4f}')
    print(f'Train scaled: mean={train_scaled.mean():.4f}, std={train_scaled.std():.4f}')
    print(f'Val   scaled: mean={val_scaled.mean():.4f}, std={val_scaled.std():.4f}')
    print(f'Test  scaled: mean={test_scaled.mean():.4f}, std={test_scaled.std():.4f}')

    out_path = os.path.join(OUT_DIR, 'standardized.npz')

    # saves standardized data in 8 arrays containing key, shape, and dtype
    # each key maps to an array containing the standardized data
    np.savez(
        out_path,
        train_data=train_scaled, train_labels=train_labels,
        val_data=val_scaled, val_labels=val_labels,
        test_data=test_scaled, test_labels=test_labels,
        mu=mu, sigma=sigma,
    )
