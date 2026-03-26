"""Data helpers for the MNIST demo.

Called from main.ein via python::data_loader::*.
Returns flattened, inverted images [10, 784] and one-hot labels [10, 10].
"""

import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SAMPLES = os.path.join(_DIR, "samples")
_WEIGHTS = os.path.join(_DIR, "weights")
_Q_WEIGHTS = os.path.join(_DIR, "..", "mnist_quantized", "weights")


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
    os.makedirs(_WEIGHTS, exist_ok=True)
    return np.zeros((784, 10), dtype=np.float32)


def ensure_weights() -> np.float32:
    """Ensure float MNIST CNN weights exist locally.

    If missing, derive them from quantized checkpoints in examples/mnist_quantized/weights:
      float_w = int8_w * scale
    Bias tensors are copied as-is.
    """
    os.makedirs(_WEIGHTS, exist_ok=True)

    targets = {
        "conv1_w.npy": (8, 1, 5, 5),
        "conv1_b.npy": (8,),
        "conv2_w.npy": (16, 8, 5, 5),
        "conv2_b.npy": (16,),
        "fc_w.npy": (256, 10),
        "fc_b.npy": (10,),
    }
    if all(os.path.exists(os.path.join(_WEIGHTS, name)) for name in targets):
        return np.float32(0.0)

    # Source quantized tensors/scales
    conv1_w_q = np.load(os.path.join(_Q_WEIGHTS, "conv1_w_q.npy")).astype(np.float32)
    conv1_w_s = float(np.load(os.path.join(_Q_WEIGHTS, "conv1_w_s.npy")).reshape(-1)[0])
    conv2_w_q = np.load(os.path.join(_Q_WEIGHTS, "conv2_w_q.npy")).astype(np.float32)
    conv2_w_s = float(np.load(os.path.join(_Q_WEIGHTS, "conv2_w_s.npy")).reshape(-1)[0])
    fc_w_q = np.load(os.path.join(_Q_WEIGHTS, "fc_w_q.npy")).astype(np.float32)
    fc_w_s = float(np.load(os.path.join(_Q_WEIGHTS, "fc_w_s.npy")).reshape(-1)[0])

    conv1_b = np.load(os.path.join(_Q_WEIGHTS, "conv1_b.npy")).astype(np.float32)
    conv2_b = np.load(os.path.join(_Q_WEIGHTS, "conv2_b.npy")).astype(np.float32)
    fc_b = np.load(os.path.join(_Q_WEIGHTS, "fc_b.npy")).astype(np.float32)

    conv1_w = (conv1_w_q * conv1_w_s).astype(np.float32)
    conv2_w = (conv2_w_q * conv2_w_s).astype(np.float32)
    fc_w = (fc_w_q * fc_w_s).astype(np.float32)

    # Materialize only missing files; do not overwrite trained classifier weights.
    if not os.path.exists(os.path.join(_WEIGHTS, "conv1_w.npy")):
        np.save(os.path.join(_WEIGHTS, "conv1_w.npy"), conv1_w)
    if not os.path.exists(os.path.join(_WEIGHTS, "conv1_b.npy")):
        np.save(os.path.join(_WEIGHTS, "conv1_b.npy"), conv1_b)
    if not os.path.exists(os.path.join(_WEIGHTS, "conv2_w.npy")):
        np.save(os.path.join(_WEIGHTS, "conv2_w.npy"), conv2_w)
    if not os.path.exists(os.path.join(_WEIGHTS, "conv2_b.npy")):
        np.save(os.path.join(_WEIGHTS, "conv2_b.npy"), conv2_b)
    if not os.path.exists(os.path.join(_WEIGHTS, "fc_w.npy")):
        np.save(os.path.join(_WEIGHTS, "fc_w.npy"), fc_w)
    if not os.path.exists(os.path.join(_WEIGHTS, "fc_b.npy")):
        np.save(os.path.join(_WEIGHTS, "fc_b.npy"), fc_b)
    if not os.path.exists(os.path.join(_WEIGHTS, "W.npy")):
        # Compatibility with older tooling that expects this filename.
        np.save(os.path.join(_WEIGHTS, "W.npy"), fc_w)

    # Sanity-check expected shapes.
    for name, shp in targets.items():
        arr = np.load(os.path.join(_WEIGHTS, name))
        if tuple(arr.shape) != tuple(shp):
            raise ValueError("ensure_weights: %s shape %s != expected %s" % (name, arr.shape, shp))

    return np.float32(0.0)

