"""PGM image loader.

Returns a float32 numpy array shaped (1, 1, h, w) with values in [0, 1].

EINLANG_MNIST_INPUT overrides the path argument when set.
"""
import os

import numpy as np


def load(path: str) -> np.ndarray:
    actual = os.environ.get("EINLANG_MNIST_INPUT", path)
    with open(actual, "rb") as f:
        assert f.readline().strip() == b"P5", "only binary PGM (P5) supported"
        wh = f.readline().strip().split()
        w, h = int(wh[0]), int(wh[1])
        maxval = int(f.readline().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = data.reshape((h, w)).astype(np.float32) / max(1, maxval)
    return img.reshape(1, 1, h, w)
