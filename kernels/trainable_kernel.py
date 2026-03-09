"""
Checked by Bank
2026-03-09

Trainable Quantum Kernel via Kernel-Target Alignment (QKTA).

This is a custom trainable fidelity quantum kernel. It uses a parameterized
feature map U(x; θ), where θ is optimized to maximize the Kernel-Target
Alignment (KTA) with the training labels:

    A(K_θ, YYᵀ) = Tr(K_θ · YYᵀ) / sqrt(Tr(K_θ²) · Tr((YYᵀ)²))

The learned kernel is then evaluated as a standard fidelity quantum kernel
using the optimized parameters.

References:
    Hubregtsen, T. et al. (2022). Training Quantum Embedding Kernels on
    Near-Term Quantum Computers. Physical Review A, 106, 042431.
    https://arxiv.org/abs/2105.02276

    Qiskit QuantumKernelTrainer:
    https://qiskit-community.github.io/qiskit-machine-learning/tutorials/08_quantum_kernel_trainer.html
"""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import sys
from pathlib import Path

# Allow direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .base import QuantumKernel
    from benchmark.metrics import analyze_circuit_resources
except ImportError:
    from kernels.base import QuantumKernel
    from benchmark.metrics import analyze_circuit_resources


def _build_trainable_circuit(
    x: np.ndarray, theta: np.ndarray, n_qubits: int, reps: int = 1
) -> QuantumCircuit:
    """Build a custom parameterized feature-map circuit U(x; theta)."""
    qc = QuantumCircuit(n_qubits)
    n_params_per_rep = n_qubits

    for r in range(reps):
        for i in range(n_qubits):
            angle = x[i % len(x)] + theta[r * n_params_per_rep + (i % n_params_per_rep)]
            qc.ry(angle, i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc


class TrainableKernel(QuantumKernel):
    """Custom trainable fidelity quantum kernel optimized via KTA."""

    def __init__(
        self,
        n_qubits: int,
        reps: int = 1,
        shots: int = 1024,
        seed: int = 42,
        max_iter: int = 50,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)
        self.reps = reps
        self.max_iter = max_iter
        self._n_params = n_qubits * reps
        self._theta = np.zeros(self._n_params, dtype=float)
        self._backend = AerSimulator(seed_simulator=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_qubits:
            raise ValueError(
                f"X must have {self.n_qubits} features, got {X.shape[1]}"
            )

        if Y is None:
            return X, None

        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if Y.shape[1] != self.n_qubits:
            raise ValueError(
                f"Y must have {self.n_qubits} features, got {Y.shape[1]}"
            )

        return X, Y

    def _overlap_circuit(
        self, x: np.ndarray, x_prime: np.ndarray, theta: np.ndarray
    ) -> QuantumCircuit:
        """Build overlap circuit U†(x';θ) U(x;θ) for fidelity estimation."""
        phi_x = _build_trainable_circuit(x, theta, self.n_qubits, self.reps)
        phi_xp = _build_trainable_circuit(x_prime, theta, self.n_qubits, self.reps)

        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.compose(phi_x, inplace=True)
        qc.compose(phi_xp.inverse(), inplace=True)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def _build_K(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Build the symmetric training kernel matrix K_theta(X, X)."""
        n = len(X)
        K = np.zeros((n, n), dtype=float)

        circuits = []
        indices = []

        for i in range(n):
            K[i, i] = 1.0  # exact fidelity diagonal
            for j in range(i + 1, n):
                qc = self._overlap_circuit(X[i], X[j], theta)
                circuits.append(qc)
                indices.append((i, j))

        if circuits:
            t_circuits = transpile(circuits, self._backend, optimization_level=0)
            job = self._backend.run(t_circuits, shots=self.shots)
            counts_list = job.result().get_counts()

            if not isinstance(counts_list, list):
                counts_list = [counts_list]

            zero_key = "0" * self.n_qubits
            for count, (i, j) in zip(counts_list, indices):
                val = count.get(zero_key, 0) / self.shots
                K[i, j] = val
                K[j, i] = val

        return K

    @staticmethod
    def _kta(K: np.ndarray, y: np.ndarray) -> float:
        """Normalized kernel-target alignment score (higher is better)."""
        T = np.outer(y, y).astype(float)
        num = np.trace(K @ T)
        denom = np.sqrt(np.trace(K @ K) * np.trace(T @ T))
        return num / (denom + 1e-12)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainableKernel":
        """Optimize theta to maximize KTA on the training set."""
        X, _ = self._validate_inputs(X)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        rng = np.random.default_rng(self.seed)
        theta0 = rng.uniform(-np.pi, np.pi, self._n_params)

        with tqdm(
            total=self.max_iter,
            desc="  QKTA fit",
            unit="iter",
            ncols=88,
            leave=False,
        ) as pbar:

            def neg_kta(theta: np.ndarray) -> float:
                K = self._build_K(X, theta)
                kta = self._kta(K, y)
                pbar.update(1)
                pbar.set_postfix(kta=f"{kta:.4f}", refresh=True)
                return -kta

            result = minimize(
                neg_kta,
                theta0,
                method="COBYLA",
                options={"maxiter": self.max_iter, "rhobeg": 0.5},
            )

        self._theta = result.x
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute the kernel matrix using the learned theta.

        If Y is None, compute the symmetric self-kernel K(X, X).
        Otherwise compute the rectangular kernel K(X, Y).
        """
        self._reset_stats()
        t0 = time.perf_counter()

        X, Y = self._validate_inputs(X, Y)
        symmetric = Y is None
        Y = X if Y is None else Y

        n, m = len(X), len(Y)
        K = np.zeros((n, m), dtype=float)

        circuits = []
        indices = []

        for i in range(n):
            start_j = i if symmetric else 0
            for j in range(start_j, m):
                if symmetric and i == j:
                    K[i, i] = 1.0
                    continue

                qc = self._overlap_circuit(X[i], Y[j], self._theta)
                circuits.append(qc)
                indices.append((i, j))

        if circuits:
            self.stats.total_shots = self.shots * len(circuits)
            self.stats.n_evaluations = len(circuits)

            # Analyze a representative overlap circuit
            res = analyze_circuit_resources(circuits[0])
            self.stats.total_depth = res["total_depth"]
            self.stats.two_qubit_depth = res["two_qubit_depth"]
            self.stats.total_gates = res["total_gates"]
            self.stats.two_qubit_count = res["two_qubit_count"]
            self.stats.one_qubit_count = res["one_qubit_count"]
            self.stats.gate_breakdown = res["gate_breakdown"]

            t_circuits = transpile(circuits, self._backend, optimization_level=0)
            job = self._backend.run(t_circuits, shots=self.shots)
            counts_list = job.result().get_counts()

            if not isinstance(counts_list, list):
                counts_list = [counts_list]

            zero_key = "0" * self.n_qubits
            for count, (i, j) in zip(counts_list, indices):
                val = count.get(zero_key, 0) / self.shots
                K[i, j] = val
                if symmetric:
                    K[j, i] = val
        else:
            # all diagonal case, e.g. n=1 symmetric
            self.stats.total_shots = 0
            self.stats.n_evaluations = 0

            # Still record representative circuit resources if possible
            example_qc = self._overlap_circuit(X[0], Y[0], self._theta)
            res = analyze_circuit_resources(example_qc)
            self.stats.total_depth = res["total_depth"]
            self.stats.two_qubit_depth = res["two_qubit_depth"]
            self.stats.total_gates = res["total_gates"]
            self.stats.two_qubit_count = res["two_qubit_count"]
            self.stats.one_qubit_count = res["one_qubit_count"]
            self.stats.gate_breakdown = res["gate_breakdown"]

        self.stats.wall_clock_seconds = time.perf_counter() - t0
        return K

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    from datasets.loader import load_dataset
    from kernels.trainable_kernel import TrainableKernel
    from classifiers.qsvm import QSVM
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(
        name="breast_cancer",
        n_samples=100,
        n_features=2,
        test_size=0.25,
        random_state=42,
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)

    # Build and train trainable kernel
    kernel = TrainableKernel(n_qubits=2, reps=2, max_iter=10, shots=1024, seed=42)
    
    print("\nTraining kernel parameters (max_iter=10)...")
    kernel.fit(X_train, y_train)

    K_train = kernel.build_kernel_matrix(X_train)
    K_test = kernel.build_kernel_matrix(X_test, X_train)

    # Kernel sanity checks
    print("\nK_train shape:", K_train.shape)
    print("K_test shape :", K_test.shape)
    print("K_train symmetric:", np.allclose(K_train, K_train.T, atol=1e-8))
    print("K_train diagonal :", np.round(np.diag(K_train), 6))
    print("K_train min/max  :", K_train.min(), K_train.max())
    print("K_test  min/max  :", K_test.min(), K_test.max())

    # Train QSVM
    model = QSVM(C=1.0)
    model.fit(K_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\ny_pred:", y_pred)
    print("y_test:", y_test)
    print(f"accuracy:  {acc:.3f}")
    print(f"precision: {prec:.3f}")
    print(f"recall:    {rec:.3f}")
    print(f"f1 score:  {f1:.3f}")

    # Resource stats
    print("\nKernel resource stats:")
    print(kernel.stats)