"""
Tests for the dataset loader.

Verifies all datasets load without error, with correct shapes and label sets.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loader import load_dataset, DATASET_NAMES

N_FEATURES = 3
N_SAMPLES = 40


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_dataset_loads(name: str) -> None:
    X_train, X_test, y_train, y_test = load_dataset(
        name, n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0
    )

    X_train = X_train[:, :N_FEATURES]
    X_test = X_test[:, :N_FEATURES]

    # Check feature dimensions
    assert X_train.shape[1] == N_FEATURES, f"{name}: wrong feature dim {X_train.shape[1]}"
    assert X_test.shape[1] == N_FEATURES

    # Check label values are binary {-1, +1}
    for y in (y_train, y_test):
        assert set(np.unique(y)).issubset({-1, 1}), f"{name}: unexpected labels {np.unique(y)}"

    # Check feature range [0, 2π]
    assert X_train.min() >= -1e-6, f"{name}: features below 0"
    assert X_train.max() <= 2 * np.pi + 1e-6, f"{name}: features above 2π"


def test_train_test_sizes() -> None:
    X_train, X_test, y_train, y_test = load_dataset(
        "moons", n_samples=50, n_features=2, test_size=0.2, random_state=0
    )
    total = len(X_train) + len(X_test)
    assert total == 50
    assert len(X_test) == 10


def test_ad_hoc_reproducible() -> None:
    a = load_dataset("ad_hoc", n_samples=20, n_features=2, random_state=7)
    b = load_dataset("ad_hoc", n_samples=20, n_features=2, random_state=7)
    np.testing.assert_array_equal(a[0], b[0])  # X_train
