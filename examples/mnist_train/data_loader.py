"""Training data loader for the MNIST training demo.

Called from main.ein via python::data_loader::{load_images, load_labels}.
Returns flattened, inverted images [10, 784] and one-hot labels [10, 10].
"""

import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SAMPLES = os.path.join(_DIR, "..", "mnist", "samples")


def load_images() -> np.ndarray:
    X = np.zeros((10, 784), dtype=np.float32)
    for i in range(10):
        with open(os.path.join(_SAMPLES, f"{i}.pgm"), "rb") as f:
            assert f.readline().strip() == b"P5", "only binary PGM (P5) supported"
            wh = f.readline().strip().split()
            w, h = int(wh[0]), int(wh[1])
            maxval = int(f.readline().strip())
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = data.reshape((h, w)).astype(np.float32) / max(1, maxval)
        X[i] = (1.0 - img).flatten()
    return X


def load_labels() -> np.ndarray:
    return np.eye(10, dtype=np.float32)


def init_weights() -> np.ndarray:
    return np.zeros((784, 10), dtype=np.float32)
