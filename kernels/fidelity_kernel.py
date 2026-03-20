from __future__ import annotations

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
from typing import Any
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
        chunk_size: int = 4096,
        backend_name: str = "aer",
        backend: Any | None = None,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name, backend=backend)
        self.feature_map = feature_map or ZZMap(n_qubits=n_qubits, reps=2)
        self.enforce_psd = enforce_psd
        self._backend = backend if backend is not None else AerSimulator(seed_simulator=seed)

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

        qc = QuantumCircuit(self.n_qubits)
        qc.compose(phi_x, inplace=True)
        qc.compose(phi_xp.inverse(), inplace=True)
        qc.measure_all()
        return qc

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
        
        if self.backend_name == "statevector":
            K = self._build_kernel_matrix_sv(X, Y, symmetric, n, m)
        elif self.backend_name == "ibm":
            K = self._build_kernel_matrix_ibm(X, Y, symmetric, n, m)
        else:
            K = self._build_kernel_matrix_aer(X, Y, symmetric, n, m)

        if self.enforce_psd and symmetric:
            K = self._project_to_psd(K)

        self.stats.wall_clock_seconds = time.perf_counter() - t0
        return K

    def _build_kernel_matrix_ibm(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool, n: int, m: int
    ) -> np.ndarray:
        """IBM Quantum Runtime path using SamplerV2 and transpilation."""
        from tqdm import tqdm
        from qiskit.circuit import ParameterVector
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        K = np.zeros((n, m), dtype=float)
        total_pairs = (n * (n + 1)) // 2 if symmetric else n * m
        chunk_size = self.chunk_size

        x_sym = ParameterVector("x", self.n_qubits)
        y_sym = ParameterVector("y", self.n_qubits)

        example_qc = self.feature_map.build(x_sym)
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        template_qc = self._overlap_circuit(x_sym, y_sym)
        tqdm.write(f"  FQK IBM: Transpiling template for {self._backend.name}...")
        t_template = transpile(template_qc, self._backend, optimization_level=3)

        x_binds = [{x_sym[k]: X[i, k] for k in range(self.n_qubits)} for i in range(n)]
        y_binds = [{y_sym[k]: Y[j, k] for k in range(self.n_qubits)} for j in range(m)]

        circuits: list[QuantumCircuit] = []
        indices: list[tuple[int, int]] = []

        self.stats.total_shots = 0
        self.stats.n_evaluations = 0

        pbar = tqdm(total=total_pairs, desc="  FQK IBM", unit="pair", ncols=88, leave=False)
        sampler = Sampler(mode=self._backend)

        def _run_chunk() -> None:
            if not circuits:
                return
            pbar.set_postfix_str("submitting...", refresh=True)
            self.stats.total_shots += self.shots * len(circuits)
            self.stats.n_evaluations += len(circuits)

            job = sampler.run(circuits, shots=self.shots)
            result = job.result()
            
            zero_key = "0" * self.n_qubits
            for pub_idx, (idx_i, idx_j) in enumerate(indices):
                pub_result = result[pub_idx]
                counts = pub_result.data.meas.get_counts()
                val = counts.get(zero_key, 0) / self.shots
                K[idx_i, idx_j] = val
                if symmetric and idx_i != idx_j:
                    K[idx_j, idx_i] = val

            pbar.update(len(circuits))
            pbar.set_postfix_str("", refresh=True)
            circuits.clear()
            indices.clear()

        for i in range(n):
            start_j = i if symmetric else 0
            for j in range(start_j, m):
                if symmetric and i == j:
                    K[i, i] = 1.0
                    pbar.update(1)
                    continue

                binds = x_binds[i].copy()
                binds.update(y_binds[j])
                qc = t_template.assign_parameters(binds)
                circuits.append(qc)
                indices.append((i, j))

                if len(circuits) >= chunk_size:
                    _run_chunk()

        if len(circuits) > 0:
            _run_chunk()

        pbar.close()
        return K

    def _build_kernel_matrix_sv(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool, n: int, m: int
    ) -> np.ndarray:
        """Fast O(N) matrix algebra path using Statevectors."""
        from tqdm import tqdm
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import Statevector

        # 1) Get single analysis of feature map encoding circuit for resource counting
        from qiskit.circuit import ParameterVector
        x_sym = ParameterVector("x", self.n_qubits)
        example_qc = self.feature_map.build(x_sym)
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        total_pairs = (n * (n + 1)) // 2 if symmetric else n * m
        self.stats.total_shots = total_pairs * self.shots
        self.stats.n_evaluations = total_pairs

        # 2) Build state encoding templates
        x_sym = ParameterVector("x", self.n_qubits)
        encode_qc = self.feature_map.build(x_sym)
        # Transpile encoding circuit
        t_encode = transpile(encode_qc, self._backend, optimization_level=0)

        def encode_dist(data: np.ndarray, desc: str) -> np.ndarray:
            svs = []
            for i in tqdm(range(len(data)), desc=desc, unit="sv", ncols=88, leave=False):
                binds = {x_sym[k]: data[i, k] for k in range(self.n_qubits)}
                qc = t_encode.assign_parameters(binds)
                svs.append(Statevector(qc).data)
            return np.array(svs)

        V_x = encode_dist(X, "  FQK SV (X)")
        V_y = V_x if symmetric else encode_dist(Y, "  FQK SV (Y)")

        # 3) Matrix multiplication for exact probabilities
        K_exact = np.abs(V_x.conj() @ V_y.T)**2
        K_exact = np.clip(K_exact, 0.0, 1.0)

        # 4) Mathematical binomial shot noise
        rng = np.random.default_rng(self.seed)
        K = rng.binomial(self.shots, K_exact) / self.shots

        # Force exact 1.0 on diagonal for symmetric
        if symmetric:
            np.fill_diagonal(K, 1.0)

        return K

    def _build_kernel_matrix_aer(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool, n: int, m: int
    ) -> np.ndarray:
        """Standard physical Qiskit Aer simulation path."""
        from tqdm import tqdm
        from qiskit.circuit import ParameterVector

        K = np.zeros((n, m), dtype=float)

        total_pairs = (n * (n + 1)) // 2 if symmetric else n * m
        chunk_size = self.chunk_size

        x_sym = ParameterVector("x", self.n_qubits)
        y_sym = ParameterVector("y", self.n_qubits)

        example_qc = self.feature_map.build(x_sym)
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        template_qc = self._overlap_circuit(x_sym, y_sym)
        t_template = transpile(template_qc, self._backend, optimization_level=0)

        x_binds = [{x_sym[k]: X[i, k] for k in range(self.n_qubits)} for i in range(n)]
        y_binds = [{y_sym[k]: Y[j, k] for k in range(self.n_qubits)} for j in range(m)]

        circuits: list[QuantumCircuit] = []
        indices: list[tuple[int, int]] = []

        # Ensure total shots start at 0 before chunks
        self.stats.total_shots = 0
        self.stats.n_evaluations = 0

        pbar = tqdm(total=total_pairs, desc="  FQK Aer", unit="pair", ncols=88, leave=False)

        def _run_chunk() -> None:
            if not circuits:
                return

            pbar.set_postfix_str("simulating...", refresh=True)

            self.stats.total_shots += self.shots * len(circuits)
            self.stats.n_evaluations += len(circuits)

            job = self._backend.run(circuits, shots=self.shots)
            counts_list = job.result().get_counts()

            if not isinstance(counts_list, list):
                counts_list = [counts_list]

            zero_key = "0" * self.n_qubits
            for count, (idx_i, idx_j) in zip(counts_list, indices):
                val = count.get(zero_key, 0) / self.shots
                K[idx_i, idx_j] = val
                if symmetric and idx_i != idx_j:
                    K[idx_j, idx_i] = val

            pbar.update(len(circuits))
            pbar.set_postfix_str("", refresh=True)
            circuits.clear()
            indices.clear()

        for i in range(n):
            start_j = i if symmetric else 0
            for j in range(start_j, m):
                if symmetric and i == j:
                    K[i, i] = 1.0
                    pbar.update(1)
                    continue

                binds = x_binds[i].copy()
                binds.update(y_binds[j])
                qc = t_template.assign_parameters(binds)
                circuits.append(qc)
                indices.append((i, j))

                if len(circuits) >= chunk_size:
                    _run_chunk()

        if len(circuits) > 0:
            _run_chunk()

        pbar.close()

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

    X_train = X_train[:, :2]
    X_test = X_test[:, :2]

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