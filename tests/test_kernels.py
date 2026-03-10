"""
Tests for quantum kernel modules.

Verifies:
  - Kernel matrix is square and has correct shape.
  - Matrix is symmetric (for train-train computation).
  - Diagonal values are close to 1 (self-similarity).
  - Matrix is positive semi-definite.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel

N_QUBITS = 2
N_SAMPLES = 6
SHOTS = 128
SEED = 0


@pytest.fixture(scope="module")
def small_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0, 2 * np.pi, (N_SAMPLES, N_QUBITS))
    y = np.array([-1, 1, -1, 1, -1, 1])
    return X, y


def assert_valid_kernel_matrix(K: np.ndarray, n: int) -> None:
    assert K.shape == (n, n), f"Expected ({n},{n}), got {K.shape}"
    assert np.allclose(K, K.T, atol=1e-6), "Kernel matrix is not symmetric"
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() > -0.05, f"Kernel not PSD: min eigenvalue {eigenvalues.min():.4f}"


# -----------------------------------------------------------------------
# FQK
# -----------------------------------------------------------------------

def test_fqk_shape_and_symmetry(small_data):
    X, _ = small_data
    kernel = FidelityKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED)
    K = kernel.build_kernel_matrix(X)
    assert_valid_kernel_matrix(K, N_SAMPLES)


def test_fqk_diagonal_near_one(small_data):
    X, _ = small_data
    kernel = FidelityKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED)
    K = kernel.build_kernel_matrix(X)
    assert np.all(np.diag(K) > 0.8), f"FQK diagonal not close to 1: {np.diag(K)}"


def test_fqk_rectangular(small_data):
    X, _ = small_data
    kernel = FidelityKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED)
    K = kernel.build_kernel_matrix(X[:4], X[4:])
    assert K.shape == (4, 2)


# -----------------------------------------------------------------------
# PQK
# -----------------------------------------------------------------------

def test_pqk_shape_and_symmetry(small_data):
    X, _ = small_data
    kernel = ProjectedKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED)
    K = kernel.build_kernel_matrix(X)
    assert_valid_kernel_matrix(K, N_SAMPLES)


def test_pqk_values_in_range(small_data):
    X, _ = small_data
    kernel = ProjectedKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED)
    K = kernel.build_kernel_matrix(X)
    assert np.all(K >= 0) and np.all(K <= 1 + 1e-6), "PQK values out of [0, 1]"


# -----------------------------------------------------------------------
# TrainableKernel (QKTA)
# -----------------------------------------------------------------------

def test_qkta_fit_and_kernel(small_data):
    X, y = small_data
    kernel = TrainableKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED, max_iter=5)
    kernel.fit(X, y)
    K = kernel.build_kernel_matrix(X)
    assert_valid_kernel_matrix(K, N_SAMPLES)


# -----------------------------------------------------------------------
# Q-FLAIR
# -----------------------------------------------------------------------

def test_qflair_fit_and_kernel(small_data):
    X, y = small_data
    kernel = QFLAIRKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED, n_layers=2)
    kernel.fit(X, y)
    K = kernel.build_kernel_matrix(X)
    assert_valid_kernel_matrix(K, N_SAMPLES)


def test_qflair_learned_architecture(small_data):
    X, y = small_data
    kernel = QFLAIRKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED, n_layers=2)
    kernel.fit(X, y)
    assert len(kernel.learned_gates) > 0, "Q-FLAIR should learn at least one gate"


# -----------------------------------------------------------------------
# Resource stats
# -----------------------------------------------------------------------

def test_resource_stats_populated(small_data):
    X, _ = small_data
    kernel = FidelityKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED)
    kernel.build_kernel_matrix(X)
    assert kernel.stats.n_qubits == N_QUBITS
    assert kernel.stats.n_evaluations > 0
    assert kernel.stats.total_shots > 0
