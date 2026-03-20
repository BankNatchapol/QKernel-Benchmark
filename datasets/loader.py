"""
Checked by Bank
2026-03-09
"""

from __future__ import annotations

"""
Dataset loader for QKernel-Benchmark.

This version keeps the ORIGINAL feature structure of every dataset.

Behavior
--------
1. Loads or generates the dataset.
2. Reduces to binary classification where applicable.
3. Uses official train/test split for HEPMASS.
4. Splits other datasets into train / test sets.
5. DOES NOT apply PCA.
6. DOES NOT reshape features.
7. DOES NOT scale features.

Notes
-----
- HIGGS is loaded from raw HIGGS.csv.gz.
- HEPMASS is loaded from official train/test CSV files.
- EnergyFlow keeps the original output from `ef.qg_jets.load(...)`.
  With pad=True, that is typically a 3D array: (n_samples, max_particles, 4).
"""

import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split


DATASET_NAMES = [
    "ad_hoc",
    "moons",
    "circles",
    "blobs",
    "iris",
    "wine",
    "breast_cancer",
    "higgs",
    "hepmass",
    "energyflow",
]

try:
    from datasets.download import _resolve_or_download, HIGGS_URL, HEPMASS_BASE_URL
except ImportError:
    from download import _resolve_or_download, HIGGS_URL, HEPMASS_BASE_URL


# -----------------------------------------------------------------------
# Quantum-hard ad-hoc dataset
# -----------------------------------------------------------------------


def _make_ad_hoc(
    n_samples: int,
    n_features: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic dataset based on random Pauli parity labeling.

    Labels are determined by a random binary function of products of
    feature pairs.
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 2 * np.pi, (n_samples, n_features))

    adj = rng.integers(0, 2, (n_features, n_features))
    adj = np.triu(adj, 1)

    if np.sum(adj) == 0 and n_features > 1:
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
# Download helpers
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Download helpers imported from download.py
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Helpers for large CSV datasets
# -----------------------------------------------------------------------


def _sample_rows_from_csv(
    path: str | os.PathLike,
    n_samples: int,
    *,
    header: int | None,
    random_state: int,
    chunksize: int = 100_000,
    balanced: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-safe random subsampling from a possibly huge CSV/CSV.GZ file.

    Assumes:
      - first column is label
      - remaining columns are features

    If balanced=True, attempts to return exactly n_samples//2 from each class
    before doing the final shuffle.
    """
    rng = np.random.default_rng(random_state)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    total_kept = 0
    total_seen = 0

    # Oversample by 3× when balanced, so we have room to balance later
    target = n_samples * 3 if balanced else n_samples

    reader = pd.read_csv(path, header=header, chunksize=chunksize)

    for chunk in reader:
        total_seen += len(chunk)

        remaining_needed = target - total_kept
        if remaining_needed <= 0:
            break

        take = min(
            len(chunk),
            max(1, int(np.ceil(remaining_needed * len(chunk) / total_seen))),
        )
        idx = rng.choice(len(chunk), size=take, replace=False)

        sampled = np.asarray(chunk)[idx]
        y_chunk = sampled[:, 0]
        X_chunk = np.asarray(sampled[:, 1:], dtype=np.float32)

        xs.append(X_chunk)
        ys.append(y_chunk)
        total_kept += len(sampled)

    if not xs:
        raise ValueError(f"No rows could be read from {path}")

    X = np.vstack(xs)
    y = np.concatenate(ys)

    if balanced:
        X, y = _balanced_sample(X, y, n_samples, rng)
    elif len(X) > n_samples:
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


def _balanced_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return exactly n_samples rows with a 50/50 class split.
    If a class doesn't have enough samples, it takes as many as possible
    and fills the rest from the other class.
    """
    unique_labels = np.unique(y)
    half = n_samples // 2

    chosen_X, chosen_y = [], []
    for label in unique_labels:
        mask = y == label
        idx_class = np.where(mask)[0]
        take = min(len(idx_class), half)
        chosen = rng.choice(idx_class, size=take, replace=False)
        chosen_X.append(X[chosen])
        chosen_y.append(y[chosen])

    X_bal = np.vstack(chosen_X)
    y_bal = np.concatenate(chosen_y)

    # If we're still short (one class is tiny), fill from what we have
    if len(X_bal) < n_samples:
        remaining = n_samples - len(X_bal)
        already_used = set()
        for i, label in enumerate(unique_labels):
            already_used.update(np.where(y == label)[0][:min(half, np.sum(y == label))])
        # Fall back: just shuffle and truncate
        idx_all = rng.permutation(len(X_bal))
        X_bal = X_bal[idx_all]
        y_bal = y_bal[idx_all]
    else:
        # Shuffle the balanced set
        idx_shuf = rng.permutation(len(X_bal))
        X_bal = X_bal[idx_shuf[:n_samples]]
        y_bal = y_bal[idx_shuf[:n_samples]]

    return X_bal, y_bal


# -----------------------------------------------------------------------
# High Energy Physics (HEP) datasets
# -----------------------------------------------------------------------


def _load_higgs(
    n_samples: int,
    random_state: int,
    higgs_path: str | os.PathLike | None = None,
    chunksize: int = 100_000,
    auto_download: bool = False,
    overwrite_download: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load HIGGS from raw HIGGS.csv.gz.

    Format:
      - no header
      - col 0 = label
      - cols 1: = 28 features
    """
    path = _resolve_or_download(
        higgs_path,
        env_var="HIGGS_PATH",
        default_relative="HIGGS/HIGGS.csv.gz",
        download_url=HIGGS_URL,
        filename_hint="HIGGS.csv.gz",
        auto_download=auto_download,
        overwrite=overwrite_download,
    )
    return _sample_rows_from_csv(
        path,
        n_samples=n_samples,
        header=None,
        random_state=random_state,
        chunksize=chunksize,
        balanced=True,  # Enforce 50/50 class balance
    )


def _load_hepmass_split(
    variant: str = "all",
    train_path: str | os.PathLike | None = None,
    test_path: str | os.PathLike | None = None,
    *,
    chunksize: int = 100_000,
    n_train_samples: int | None = None,
    n_test_samples: int | None = None,
    random_state: int = 42,
    auto_download: bool = False,
    overwrite_download: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load HEPMASS using its official train/test split.

    Note:
      The files on the server are plain .csv, not .csv.gz.

    Variants
    --------
    - "1000"
    - "not1000"
    - "all"

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    valid_variants = {"1000", "not1000", "all"}
    if variant not in valid_variants:
        raise ValueError(f"Invalid HEPMASS variant '{variant}'. Choose from {valid_variants}.")

    train_filename = f"{variant}_train.csv"
    test_filename = f"{variant}_test.csv"

    train_url = f"{HEPMASS_BASE_URL}/{train_filename}"
    test_url = f"{HEPMASS_BASE_URL}/{test_filename}"
    train_file = _resolve_or_download(
        train_path,
        env_var="HEPMASS_TRAIN_PATH",
        default_relative=f"HEPMASS/{train_filename}",
        download_url=train_url,
        filename_hint=train_filename,
        auto_download=auto_download,
        overwrite=overwrite_download,
    )

    test_file = _resolve_or_download(
        test_path,
        env_var="HEPMASS_TEST_PATH",
        default_relative=f"HEPMASS/{test_filename}",
        download_url=test_url,
        filename_hint=test_filename,
        auto_download=auto_download,
        overwrite=overwrite_download,
    )

    if n_train_samples is None:
        train_df = pd.read_csv(train_file)
        y_train_raw = train_df.iloc[:, 0].to_numpy()
        X_train_raw = train_df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        rng_tr = np.random.default_rng(random_state)
        X_train, y_train = _balanced_sample(X_train_raw, y_train_raw, len(y_train_raw), rng_tr)
    else:
        X_train, y_train = _sample_rows_from_csv(
            train_file,
            n_samples=n_train_samples,
            header=0,
            random_state=random_state,
            chunksize=chunksize,
            balanced=True,
        )

    if n_test_samples is None:
        test_df = pd.read_csv(test_file)
        y_test_raw = test_df.iloc[:, 0].to_numpy()
        X_test_raw = test_df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        rng_te = np.random.default_rng(random_state + 1)
        X_test, y_test = _balanced_sample(X_test_raw, y_test_raw, len(y_test_raw), rng_te)
    else:
        X_test, y_test = _sample_rows_from_csv(
            test_file,
            n_samples=n_test_samples,
            header=0,
            random_state=random_state + 1,
            chunksize=chunksize,
            balanced=True,
        )

    return X_train, X_test, y_train, y_test


def _load_energyflow(
    n_samples: int,
    *,
    cache_dir: str | os.PathLike,
    generator: str = "pythia",
    with_bc: bool = False,
    pad: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load EnergyFlow qg_jets and KEEP the original feature shape.

    Returns
    -------
    X:
      - if pad=True: usually shape (n_samples, max_particles, 4)
      - if pad=False: object array of variable-length jets
    y:
      - shape (n_samples,)
    """
    try:
        import energyflow as ef
    except Exception as e:
        raise ImportError(
            "Failed to import energyflow. "
            "Install it in the current environment with:\n"
            "    python -m pip install energyflow\n"
            f"Original error: {e}"
        ) from e

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    X, y = ef.qg_jets.load(
        num_data=n_samples,
        pad=pad,
        ncol=4,
        generator=generator,
        with_bc=with_bc,
        cache_dir=str(cache_dir),
    )

    y = np.asarray(y).ravel()
    return X, y


# -----------------------------------------------------------------------
# Main loader
# -----------------------------------------------------------------------


def load_dataset(
    name: str,
    n_samples: int = 100,
    n_features: int = 4,  # kept for compatibility, not used except ad_hoc generation
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    higgs_path: str | os.PathLike | None = None,
    hepmass_variant: str = "all",
    hepmass_train_path: str | os.PathLike | None = None,
    hepmass_test_path: str | os.PathLike | None = None,
    chunksize: int = 100_000,
    energyflow_cache_dir: str | os.PathLike = Path(__file__).resolve().parent / "ENERGYFLOW",
    energyflow_generator: str = "pythia",
    energyflow_with_bc: bool = False,
    energyflow_pad: bool = True,
    auto_download: bool = False,
    overwrite_download: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset and KEEP the original feature structure.

    Returns
    -------
    X_train, X_test, y_train, y_test
        No PCA, no reshape, no scaling.
    """
    name = name.lower()

    # -------------------------------------------------------------------
    # HEPMASS uses official train/test split, so handle it separately
    # -------------------------------------------------------------------
    if name == "hepmass":
        n_test_samples = max(1, int(round(n_samples * test_size)))
        n_train_samples = max(1, n_samples - n_test_samples)
        
        X_train, X_test, y_train, y_test = _load_hepmass_split(
            variant=hepmass_variant,
            train_path=hepmass_train_path,
            test_path=hepmass_test_path,
            chunksize=chunksize,
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            random_state=random_state,
            auto_download=auto_download,
            overwrite_download=overwrite_download,
        )

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train).ravel()
        y_test = np.asarray(y_test).ravel()

        unique = np.unique(np.concatenate([y_train, y_test]))
        if len(unique) != 2:
            raise ValueError(f"Expected binary labels for HEPMASS, got {unique}")

        y_train = np.where(y_train == unique[0], -1, 1).astype(int)
        y_test = np.where(y_test == unique[0], -1, 1).astype(int)

        return X_train, X_test, y_train, y_test

    # -------------------------------------------------------------------
    # Load raw data for all other datasets
    # -------------------------------------------------------------------
    if name == "ad_hoc":
        X, y = _make_ad_hoc(n_samples, n_features, random_state)

    elif name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)

    elif name == "circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=0.1,
            factor=0.4,
            random_state=random_state,
        )

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

    elif name == "higgs":
        X, y = _load_higgs(
            n_samples=n_samples,
            random_state=random_state,
            higgs_path=higgs_path,
            chunksize=chunksize,
            auto_download=auto_download,
            overwrite_download=overwrite_download,
        )

    elif name == "energyflow":
        # Load extra samples so _balanced_sample has enough of each jet type
        X, y = _load_energyflow(
            n_samples=n_samples * 4,
            cache_dir=energyflow_cache_dir,
            generator=energyflow_generator,
            with_bc=energyflow_with_bc,
            pad=energyflow_pad,
        )

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {DATASET_NAMES}")

    y = np.asarray(y).ravel()

    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(
            f"Expected binary labels, but got {len(unique)} unique labels: {unique}"
        )
    y = np.where(y == unique[0], -1, 1).astype(int)

    if name not in ("ad_hoc", "higgs", "energyflow") and len(y) > n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # Enforce 50/50 balance for HEP datasets before splitting
    if name in ("higgs", "energyflow"):
        rng_bal = np.random.default_rng(random_state)
        X, y = _balanced_sample(X, y, n_samples, rng_bal)

    _, counts = np.unique(y, return_counts=True)
    can_stratify = np.all(counts >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Testing dataset loading...")

    tests = [
        ("higgs", dict()),
        # ("hepmass", dict(hepmass_variant="all")),
        # ("energyflow", dict()),
    ]

    for dataset_name, extra_kwargs in tests:
        print(f"\nLoading {dataset_name}...")
        try:
            X_train, X_test, y_train, y_test = load_dataset(
                name=dataset_name,
                n_samples=50,
                n_features=4,
                test_size=0.2,
                random_state=42,
                auto_download=False,
                overwrite_download=False,
                **extra_kwargs,
            )
            print(X_train[0])
            print(f"  X_train shape: {np.shape(X_train)}")
            print(f"  y_train shape: {y_train.shape} (unique: {np.unique(y_train)})")
            if isinstance(X_train, np.ndarray) and X_train.dtype != object:
                print(f"  X_train dtype : {X_train.dtype}")
            else:
                print("  X_train dtype : object / variable-length")
            print(f"  X_test shape : {np.shape(X_test)}")
        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")