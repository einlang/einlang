"""scikit-learn diabetes train/test split for Einlang examples.

Called from .ein files via python::diabetes_data::*.
"""

from functools import lru_cache

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


@lru_cache(maxsize=1)
def _load_diabetes_train_test_cached():
    """Return the canonical diabetes split used by the Einlang example."""
    X, y = load_diabetes(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    return X_train, y_train, X_test, y_test


def load_train_features():
    return _load_diabetes_train_test_cached()[0]


def load_train_targets():
    return _load_diabetes_train_test_cached()[1]


def load_test_features():
    return _load_diabetes_train_test_cached()[2]


def load_test_targets():
    return _load_diabetes_train_test_cached()[3]
