"""
Checked by Bank
2026-03-07

Dataset loader for QKernel-Benchmark.

Provides a unified ``load_dataset`` function that:
  1. Loads or generates the dataset.
  2. Reduces to binary classification (first 2 classes).
  3. PCA-reduces to ``n_features`` dimensions (= number of qubits).
  4. MinMax-scales features to [0, 2π].
  5. Splits into train / test sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    load_iris,
    load_wine,
    load_breast_cancer,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATASET_NAMES = [
    "ad_hoc",
    "moons",
    "circles",
    "blobs",
    "iris",
    "wine",
    "breast_cancer",
]


# -----------------------------------------------------------------------
# Quantum-hard ad-hoc dataset
# -----------------------------------------------------------------------


def _make_ad_hoc(
    n_samples: int, n_features: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic dataset based on random Pauli parity labeling.

    Labels are determined by a random binary function of products of
    feature pairs — hard to learn classically but structured enough that
    a quantum kernel may exploit the parity.

    Reference:
        Havlíček et al. (2019), Supplementary Material, Section S1.
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 2 * np.pi, (n_samples, n_features))

    # Random adjacency matrix for pair interactions
    adj = rng.integers(0, 2, (n_features, n_features))
    adj = np.triu(adj, 1)

    # Ensure at least one active pair
    if np.sum(adj) == 0:
        i, j = 0, 1 if n_features > 1 else (0, 0)
        if n_features > 1:
            adj[0, 1] = 1

    scores = np.zeros(n_samples, dtype=float)
    for k in range(n_samples):
        s = sum(
            X[k, i] * X[k, j]
            for i in range(n_features)
            for j in range(i + 1, n_features)
            if adj[i, j]
        )
        scores[k] = s

    threshold = np.median(scores)
    y = np.where(scores > threshold, 1, -1).astype(int)
    return X, y


# -----------------------------------------------------------------------
# Main loader
# -----------------------------------------------------------------------

def load_dataset(
    name: str,
    n_samples: int = 100,
    n_features: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess a dataset for quantum kernel experiments.

    Parameters
    ----------
    name : str
        One of DATASET_NAMES.
    n_samples : int
        Total number of samples (for synthetic datasets).
    n_features : int
        Number of features after PCA (= number of qubits).
    test_size : float
        Fraction of data used for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Features scaled to [0, 2π] and binary labels {-1, +1}.
    """
    name = name.lower()

    # --- Load raw data ---
    if name == "ad_hoc":
        X, y = _make_ad_hoc(n_samples, n_features, random_state)
    elif name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.4, random_state=random_state)
    elif name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=2, random_state=random_state)
    elif name == "iris":
        data = load_iris()
        mask = data.target < 2
        X, y = data.data[mask], data.target[mask]
    elif name == "wine":
        data = load_wine()
        mask = data.target < 2
        X, y = data.data[mask], data.target[mask]
    elif name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {DATASET_NAMES}")

    # --- Binary labels: map to {-1, +1} ---
    unique = np.unique(y)
    y = np.where(y == unique[0], -1, 1)

    # --- Subsample for synthetic datasets (already sized) ---
    if name not in ("ad_hoc",) and len(X) > n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    # --- PCA to n_features dimensions ---
    if X.shape[1] > n_features:
        pca = PCA(n_components=n_features, random_state=random_state)
        X = pca.fit_transform(X)
    elif X.shape[1] < n_features:
        # Zero-pad if fewer features than requested qubits
        pad = np.zeros((len(X), n_features - X.shape[1]))
        X = np.hstack([X, pad])

    # --- MinMax scale to [0, π] ---
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    # --- Train / test split ---
    # Only stratify when every class has at least 2 members; otherwise a
    # small or imbalanced dataset would raise a ValueError from sklearn.
    class_counts = np.bincount(y + 1)  # shift -1/+1 → 0/2 → use as index
    can_stratify = bool(np.all(class_counts[class_counts > 0] >= 2))
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load ad_hoc dataset
    X_train, X_test, y_train, y_test = load_dataset(
        name="ad_hoc",
        n_samples=200,
        n_features=2,   # use 2 so it can be plotted directly
        test_size=0.2,
        random_state=42,
    )

    # Combine train + test for visualization
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])

    # Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        label="Class -1",
        alpha=0.8,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Class +1",
        alpha=0.8,
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("ad_hoc Dataset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()