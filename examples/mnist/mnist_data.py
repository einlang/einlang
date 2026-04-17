"""MNIST data generator for Einlang training.

Provides functions to load and preprocess MNIST/digits data for training.
Called from .ein files via python::mnist_data::*.
"""

from functools import lru_cache

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

@lru_cache(maxsize=1)
def _load_mnist_train_test_cached():
    """Load digits dataset and split into train/test sets.

    Returns:
        train_images: (1437, 1, 8, 8) - normalized to [0,1]
        train_labels: (1437, 10) - one-hot encoded
        test_images: (360, 1, 8, 8) - normalized to [0,1]
        test_labels: (360, 10) - one-hot encoded
    """
    # Load the dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Normalize pixel values to [0, 1]
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Reshape to (N, 1, 8, 8) for CNN input
    X_reshaped = X_normalized.reshape(-1, 1, 8, 8)

    # Convert labels to one-hot encoding
    y_onehot = np.eye(10)[y]

    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_onehot, test_size=0.2, random_state=42, stratify=y
    )

    return X_train.astype(np.float32), y_train.astype(np.float32), \
           X_test.astype(np.float32), y_test.astype(np.float32)

def load_mnist_train_test():
    return _load_mnist_train_test_cached()

def load_train_images():
    return _load_mnist_train_test_cached()[0]

def load_train_labels():
    return _load_mnist_train_test_cached()[1]

def load_test_images():
    return _load_mnist_train_test_cached()[2]

def load_test_labels():
    return _load_mnist_train_test_cached()[3]

def load_single_batch(batch_size=32, digit=None):
    """Load a single batch of data for training.

    Args:
        batch_size: Number of samples to return
        digit: If specified, return only samples of this digit (0-9)

    Returns:
        images: (batch_size, 1, 8, 8)
        labels: (batch_size, 10) one-hot encoded
    """
    digits = load_digits()
    X, y = digits.data, digits.target

    # Filter by digit if specified
    if digit is not None:
        mask = y == digit
        X, y = X[mask], y[mask]

    # Normalize and reshape
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_reshaped = X_normalized.reshape(-1, 1, 8, 8)
    y_onehot = np.eye(10)[y]

    # Random sample
    indices = np.random.choice(len(X_reshaped), batch_size, replace=True)
    return X_reshaped[indices].astype(np.float32), y_onehot[indices].astype(np.float32)

def generate_synthetic_mnist_like(batch_size=100, image_size=8):
    """Generate synthetic data that resembles MNIST patterns.

    Creates simple geometric patterns for each digit class.
    Useful for testing and debugging.

    Returns:
        images: (batch_size, 1, image_size, image_size)
        labels: (batch_size, 10) one-hot encoded
    """
    images = []
    labels = []

    for _ in range(batch_size):
        digit = np.random.randint(0, 10)
        img = np.zeros((image_size, image_size), dtype=np.float32)

        # Create simple patterns for each digit
        if digit == 0:
            # Circle
            center = image_size // 2
            radius = image_size // 3
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center)**2 + (y - center)**2 <= radius**2
            img[mask] = 1.0
        elif digit == 1:
            # Vertical line
            img[:, image_size//2] = 1.0
        elif digit == 2:
            # Diagonal line
            for i in range(image_size):
                if i < image_size//2:
                    img[i, i] = 1.0
                else:
                    img[i, image_size-1-i] = 1.0
        elif digit == 3:
            # Plus sign
            center = image_size // 2
            img[center, :] = 1.0
            img[:, center] = 1.0
        elif digit == 4:
            # X shape
            for i in range(image_size):
                img[i, i] = 1.0
                img[i, image_size-1-i] = 1.0
        elif digit == 5:
            # Square
            img[1:-1, 1] = 1.0
            img[1:-1, -2] = 1.0
            img[1, 1:-1] = 1.0
            img[-2, 1:-1] = 1.0
        elif digit == 6:
            # Triangle
            for i in range(image_size//2):
                start = image_size//2 - i
                end = image_size//2 + i + 1
                img[image_size//2 + i, start:end] = 1.0
        elif digit == 7:
            # Right angle
            img[:image_size//2, 0] = 1.0
            img[0, :image_size//2] = 1.0
        elif digit == 8:
            # Diamond
            center = image_size // 2
            for i in range(center + 1):
                width = center - i
                img[center - i, center - width:center + width + 1] = 1.0
                img[center + i, center - width:center + width + 1] = 1.0
        elif digit == 9:
            # Star pattern
            center = image_size // 2
            img[center, :] = 1.0
            img[:, center] = 1.0
            for i in range(0, image_size, 2):
                img[i, i] = 1.0
                img[i, image_size-1-i] = 1.0

        # Add some noise
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)

        images.append(img.reshape(1, image_size, image_size))
        labels.append(digit)

    images = np.array(images)
    labels_onehot = np.eye(10)[labels]

    return images.astype(np.float32), labels_onehot.astype(np.float32)
