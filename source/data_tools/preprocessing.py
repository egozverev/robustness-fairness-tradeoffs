import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

import torch

from sklearn.model_selection import train_test_split


def preprocess_data(X_train, X_test):
    """
    Scale data, add a column with ones.
    :param X_train: train data
    :param X_test: test data
    :return:
    """
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)

    X_test = scaler.transform(X_test)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
    return (X_train, X_test)


def prepare_cv_datasets(x, y, groups, n_splits=5, random_state=42):
    """
    Yields preprocessed dataests as torch tensors
    :param x: features
    :param y: labels
    :param groups: groups binary labels
    :param n_splits: number of splits for (x, y, groups)
    :return:
    """
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)
    for train_index, test_index in cv.split(x):
        X_train = x[train_index]
        y_train = y[train_index]
        X_test = x[test_index]
        y_test = y[test_index]
        groups_train = groups[train_index]
        groups_test = groups[test_index]
        X_train, X_test = preprocess_data(X_train, X_test)
        yield torch.Tensor(X_train), torch.Tensor(y_train), torch.Tensor(X_test), \
              torch.Tensor(y_test), torch.Tensor(groups_train), torch.Tensor(groups_test)


def get_train_val_test_split(x, y, groups, random_state=42):
    """
    Get train, val and test split for features, labels and groups
    :param x: features
    :param y: labels
    :param groups: groups binary labels
    """
    X_train, y_train, X_test, y_test, groups_train, groups_test = next(prepare_cv_datasets(x, y, groups, 1, random_state))
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_train, y_train, groups_train, test_size=0.2, random_state=random_state)
    data_split = {
        "train": (X_train, y_train, groups_train),
        "val": (X_val, y_val, groups_val),
        "test": (X_test, y_test, groups_test)
    }
    return data_split
