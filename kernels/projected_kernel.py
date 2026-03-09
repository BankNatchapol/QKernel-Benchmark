"""
Checked by Bank
2026-03-09
"""

"""
Projected Quantum Kernel (PQK).

Encodes each input into a quantum state, extracts single-qubit reduced-state
information via local Pauli expectations <X>, <Y>, <Z> for each qubit,
concatenates these into a classical projection vector, and applies a classical
RBF kernel to the resulting vectors.

In this implementation, the projected features are computed exactly from the
statevector by taking 1-qubit reduced density matrices and converting them to
Bloch-vector coordinates. Thus, no shot-based measurement circuits are executed.

Formula:
    k^PQ(x, x') = exp(-gamma * ||b(x) - b(x')||^2)

where b(x) is the concatenated Bloch-vector representation of all 1-qubit
reduced density matrices:
    b(x) = [<X_0>, <Y_0>, <Z_0>, ..., <X_{n-1}>, <Y_{n-1}>, <Z_{n-1}>]

Reference:
    IBM Quantum tutorial:
    https://quantum.cloud.ibm.com/docs/en/tutorials/projected-quantum-kernels
"""

import time
import numpy as np
from qiskit.quantum_info import Statevector, partial_trace

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


class ProjectedKernel(QuantumKernel):
    """Projected Quantum Kernel using an RBF kernel over local Pauli expectations."""

    def __init__(
        self,
        n_qubits: int,
        feature_map: FeatureMap | None = None,
        gamma: float = 1.0,
        shots: int = 1024,
        seed: int = 42,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)
        self.feature_map = feature_map or ZZMap(n_qubits=n_qubits, reps=2)
        self.gamma = gamma

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

    def _bloch_vector(self, x: np.ndarray) -> np.ndarray:
        """Return the 3n-dimensional Bloch vector [<X>, <Y>, <Z>] per qubit."""
        qc = self.feature_map.build(x)
        sv = Statevector(qc)

        bloch = []
        for qubit in range(self.n_qubits):
            # Trace out all qubits except the current one to get its 1-qubit RDM.
            qubits_to_trace = [q for q in range(self.n_qubits) if q != qubit]
            rho = partial_trace(sv, qubits_to_trace)

            # For a 1-qubit density matrix rho:
            # <X> = 2 Re(rho[0,1])
            # <Y> = 2 Im(rho[1,0]) = -2 Im(rho[0,1])
            # <Z> = rho[0,0] - rho[1,1]
            data = rho.data
            ex = 2.0 * data[0, 1].real
            ey = 2.0 * data[1, 0].imag
            ez = (data[0, 0] - data[1, 1]).real

            bloch.extend([ex, ey, ez])

        return np.array(bloch, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute the projected quantum kernel matrix.

        Notes
        -----
        In this exact-statevector implementation, stats.n_evaluations counts
        the number of unique projected-feature computations (one per sample
        whose Bloch vector is extracted), not literal shot-based backend
        measurement circuits.
        """
        self._reset_stats()
        t0 = time.perf_counter()

        X, Y = self._validate_inputs(X, Y)
        symmetric = Y is None
        Y = X if Y is None else Y

        # Compute projected feature vectors
        X_bloch = np.array([self._bloch_vector(x) for x in X], dtype=float)
        Y_bloch = X_bloch if symmetric else np.array(
            [self._bloch_vector(y) for y in Y], dtype=float
        )

        # In this statevector-based implementation:
        # - one evaluation corresponds to one sample's Bloch-vector extraction
        # - no shot-based circuits are executed
        self.stats.n_evaluations = len(X) if symmetric else (len(X) + len(Y))
        self.stats.total_shots = 0

        # Track representative circuit resources for one feature-map circuit
        example_qc = self.feature_map.build(X[0])
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        # Classical RBF kernel over projected Bloch vectors
        n, m = len(X), len(Y)
        K = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                diff = X_bloch[i] - Y_bloch[j]
                K[i, j] = np.exp(-self.gamma * np.dot(diff, diff))

        self.stats.wall_clock_seconds = time.perf_counter() - t0
        return K

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    from datasets.loader import load_dataset
    from kernels.projected_kernel import ProjectedKernel
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

    # Build projected kernel matrices
    kernel = ProjectedKernel(n_qubits=2, shots=1024, seed=42)

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