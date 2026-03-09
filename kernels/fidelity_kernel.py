"""
Checked by Bank
2026-03-09
"""

"""
Fidelity Quantum Kernel (FQK).

Computes k(x, x') = |<ϕ(x)|ϕ(x')>|² using the adjoint overlap circuit:
    U†(x') · U(x)|0⟩  →  measure probability of all-zeros outcome.

References:
    Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. Nature, 567, 209–212.
    https://doi.org/10.1038/s41586-019-0980-2
"""

import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import sys
from pathlib import Path

# Allow direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from feature_maps.base import FeatureMap
    from feature_maps.zz_map import ZZMap
    from .base import QuantumKernel
    from benchmark.metrics import analyze_circuit_resources
except ImportError:
    from feature_maps.base import FeatureMap
    from feature_maps.zz_map import ZZMap
    from kernels.base import QuantumKernel
    from benchmark.metrics import analyze_circuit_resources


class FidelityKernel(QuantumKernel):
    """Fidelity Quantum Kernel via adjoint overlap circuit."""

    def __init__(
        self,
        n_qubits: int,
        feature_map: FeatureMap | None = None,
        shots: int = 1024,
        seed: int = 42,
        enforce_psd: bool = True,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)
        self.feature_map = feature_map or ZZMap(n_qubits=n_qubits, reps=2)
        self.enforce_psd = enforce_psd
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

    def _overlap_circuit(self, x: np.ndarray, x_prime: np.ndarray) -> QuantumCircuit:
        """Build overlap circuit U†(x') U(x) and measure all qubits."""
        phi_x = self.feature_map.build(x)
        phi_xp = self.feature_map.build(x_prime)

        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.compose(phi_x, inplace=True)
        qc.compose(phi_xp.inverse(), inplace=True)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def _project_to_psd(self, K: np.ndarray) -> np.ndarray:
        """Project a symmetric matrix to the nearest PSD matrix by clipping eigenvalues."""
        K = 0.5 * (K + K.T)
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals_clipped = np.clip(eigvals, 0.0, None)
        K_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        K_psd = 0.5 * (K_psd + K_psd.T)
        return K_psd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        self._reset_stats()
        t0 = time.perf_counter()

        X, Y = self._validate_inputs(X, Y)
        symmetric = Y is None
        Y = X if Y is None else Y

        n, m = len(X), len(Y)
        K = np.zeros((n, m), dtype=float)

        circuits: list[QuantumCircuit] = []
        indices: list[tuple[int, int]] = []

        for i in range(n):
            start_j = i if symmetric else 0
            for j in range(start_j, m):
                # In exact theory, k(x, x) = 1.
                # Avoid wasting shots and injecting sampling noise on the diagonal.
                if symmetric and i == j:
                    K[i, i] = 1.0
                    continue

                qc = self._overlap_circuit(X[i], Y[j])
                circuits.append(qc)
                indices.append((i, j))

        # Resource analysis on a representative overlap circuit, not just the feature map.
        if len(circuits) > 0:
            res = analyze_circuit_resources(circuits[0])
            self.stats.total_depth = res["total_depth"]
            self.stats.two_qubit_depth = res["two_qubit_depth"]
            self.stats.total_gates = res["total_gates"]
            self.stats.two_qubit_count = res["two_qubit_count"]
            self.stats.one_qubit_count = res["one_qubit_count"]
            self.stats.gate_breakdown = res["gate_breakdown"]

            self.stats.total_shots = self.shots * len(circuits)
            self.stats.n_evaluations = len(circuits)

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

        if symmetric and self.enforce_psd:
            K = self._project_to_psd(K)
            np.fill_diagonal(K, 1.0)

        self.stats.wall_clock_seconds = time.perf_counter() - t0
        return K

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    from datasets.loader import load_dataset
    from kernels.fidelity_kernel import FidelityKernel
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

    # Build fidelity kernel matrices
    kernel = FidelityKernel(n_qubits=2, shots=1024, seed=42)

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