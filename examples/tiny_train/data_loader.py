"""Synthetic data for the tiny training demo (32×16 features, 2 classes).

Class is sign of the first feature after standardizing columns (linearly separable).
Called from main.ein via python::data_loader::{load_features, load_labels, init_weights}.
"""

import numpy as np

_N = 32
_F = 16
_SEED = 42


def load_features() -> np.ndarray:
    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal((_N, _F)).astype(np.float32)
    x -= x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    x /= s
    return x


def load_labels() -> np.ndarray:
    x = load_features()
    idx = (x[:, 0] > 0.0).astype(np.int64)
    return np.eye(2, dtype=np.float32)[idx]


def init_weights() -> np.ndarray:
    return np.zeros((_F, 2), dtype=np.float32)
