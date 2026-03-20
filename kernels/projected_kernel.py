from __future__ import annotations

import time
from typing import Any
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
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
        chunk_size: int = 4096,
        backend_name: str = "aer",
        backend: Any | None = None,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name, backend=backend)
        self.feature_map = feature_map or ZZMap(n_qubits=n_qubits, reps=2)
        self.gamma = gamma
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
        """
        self._reset_stats()
        t0 = time.perf_counter()

        X, Y = self._validate_inputs(X, Y)
        symmetric = Y is None
        Y = X if Y is None else Y

        if self.backend_name == "statevector":
            K = self._build_kernel_matrix_sv(X, Y, symmetric)
        elif self.backend_name == "ibm":
            K = self._build_kernel_matrix_ibm(X, Y, symmetric)
        else:
            K = self._build_kernel_matrix_aer(X, Y, symmetric)

        self.stats.wall_clock_seconds = time.perf_counter() - t0
        return K

    def _build_kernel_matrix_ibm(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool
    ) -> np.ndarray:
        """IBM Quantum Runtime path via measurements in X, Y, Z bases."""
        X_bloch = self._get_bloch_vectors_ibm(X, "  PQK IBM (X)")
        Y_bloch = X_bloch if symmetric else self._get_bloch_vectors_ibm(Y, "  PQK IBM (Y)")

        # Track representative circuit resources (base encoding)
        example_qc = self.feature_map.build(X[0])
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        return self._compute_rbf_kernel(X_bloch, Y_bloch)

    def _get_bloch_vectors_ibm(self, X: np.ndarray, desc: str) -> np.ndarray:
        """Estimate 3n-dimensional Bloch vectors via measurements on IBM hardware."""
        from tqdm import tqdm
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        n_samples = len(X)
        bloch_vectors = np.zeros((n_samples, 3 * self.n_qubits))

        all_circuits = []
        for x in X:
            base_qc = self.feature_map.build(x)
            # Z basis
            qc_z = base_qc.copy()
            qc_z.measure_all()
            # X basis
            qc_x = base_qc.copy()
            for i in range(self.n_qubits): qc_x.h(i)
            qc_x.measure_all()
            # Y basis
            qc_y = base_qc.copy()
            for i in range(self.n_qubits):
                qc_y.sdg(i)
                qc_y.h(i)
            qc_y.measure_all()
            all_circuits.extend([qc_x, qc_y, qc_z])

        chunk_size = self.chunk_size * 3  # 3 circuits per sample
        pbar = tqdm(total=len(all_circuits), desc=desc, unit="circ", ncols=88, leave=False)
        sampler = Sampler(mode=self._backend)

        results_counts = []
        for i in range(0, len(all_circuits), chunk_size):
            chunk = all_circuits[i : i + chunk_size]
            tqdm.write(f"  PQK IBM: Transpiling {len(chunk)} circuits for {self._backend.name}...")
            t_chunk = transpile(chunk, self._backend, optimization_level=1)
            pbar.set_postfix_str("submitting...", refresh=True)
            job = sampler.run(t_chunk, shots=self.shots)
            res = job.result()
            
            # Extract counts from V2 results
            for pub_idx in range(len(chunk)):
                pub_res = res[pub_idx]
                results_counts.append(pub_res.data.meas.get_counts())
            
            pbar.update(len(chunk))
        pbar.close()

        self.stats.total_shots += len(all_circuits) * self.shots
        self.stats.n_evaluations += len(all_circuits)

        for i in range(n_samples):
            counts_x = results_counts[3*i]
            counts_y = results_counts[3*i+1]
            counts_z = results_counts[3*i+2]
            for q in range(self.n_qubits):
                def get_exp(counts, q_idx):
                    p0, p1 = 0, 0
                    for bits, count in counts.items():
                        if bits[self.n_qubits - 1 - q_idx] == '0': p0 += count
                        else: p1 += count
                    return (p0 - p1) / self.shots
                ex = get_exp(counts_x, q)
                ey = get_exp(counts_y, q)
                ez = get_exp(counts_z, q)
                bloch_vectors[i, 3*q : 3*q+3] = [ex, ey, ez]
        return bloch_vectors

    def _build_kernel_matrix_sv(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool
    ) -> np.ndarray:
        """Exact-statevector path."""
        from tqdm import tqdm

        # Compute projected feature vectors
        X_bloch = np.array([self._bloch_vector(x) for x in tqdm(X, desc="  PQK SV (X)", unit="smpl", ncols=88, leave=False)], dtype=float)
        Y_bloch = X_bloch if symmetric else np.array(
            [self._bloch_vector(y) for y in tqdm(Y, desc="  PQK SV (Y)", unit="smpl", ncols=88, leave=False)], dtype=float
        )

        self.stats.n_evaluations = len(X) if symmetric else (len(X) + len(Y))
        self.stats.total_shots = 0

        # Track representative circuit resources
        example_qc = self.feature_map.build(X[0])
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        return self._compute_rbf_kernel(X_bloch, Y_bloch)

    def _build_kernel_matrix_aer(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool
    ) -> np.ndarray:
        """Estimated-Aer path via measurements in X, Y, Z bases."""
        X_bloch = self._get_bloch_vectors_aer(X, "  PQK Aer (X)")
        Y_bloch = X_bloch if symmetric else self._get_bloch_vectors_aer(Y, "  PQK Aer (Y)")

        # Track representative circuit resources (base encoding)
        example_qc = self.feature_map.build(X[0])
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        return self._compute_rbf_kernel(X_bloch, Y_bloch)

    def _get_bloch_vectors_aer(self, X: np.ndarray, desc: str) -> np.ndarray:
        """Estimate 3n-dimensional Bloch vectors via measurements on Aer."""
        from tqdm import tqdm
        n_samples = len(X)
        bloch_vectors = np.zeros((n_samples, 3 * self.n_qubits))
        
        all_circuits = []
        for x in X:
            base_qc = self.feature_map.build(x)
            
            # Z basis
            qc_z = base_qc.copy()
            qc_z.measure_all()
            
            # X basis
            qc_x = base_qc.copy()
            for i in range(self.n_qubits):
                qc_x.h(i)
            qc_x.measure_all()
            
            # Y basis
            qc_y = base_qc.copy()
            for i in range(self.n_qubits):
                qc_y.sdg(i)
                qc_y.h(i)
            qc_y.measure_all()
            
            all_circuits.extend([qc_x, qc_y, qc_z])
            
        chunk_size = self.chunk_size * 3
        pbar = tqdm(total=len(all_circuits), desc=desc, unit="circ", ncols=88, leave=False)
        
        results_counts = []
        for i in range(0, len(all_circuits), chunk_size):
            chunk = all_circuits[i : i + chunk_size]
            t_chunk = transpile(chunk, self._backend, optimization_level=0)
            job = self._backend.run(t_chunk, shots=self.shots)
            res = job.result().get_counts()
            if not isinstance(res, list):
                res = [res]
            results_counts.extend(res)
            pbar.update(len(chunk))
        pbar.close()
        
        self.stats.total_shots += len(all_circuits) * self.shots
        self.stats.n_evaluations += len(all_circuits)

        for i in range(n_samples):
            # chunks of 3: X, Y, Z
            counts_x = results_counts[3*i]
            counts_y = results_counts[3*i+1]
            counts_z = results_counts[3*i+2]
            
            for q in range(self.n_qubits):
                def get_exp(counts, q_idx):
                    p0 = 0
                    p1 = 0
                    # measure_all appends ' meas' to register name, 
                    # but keys are just bitstrings like '101'
                    for bits, count in counts.items():
                        # Qiskit bitstring order is [qn-1, ..., q0]
                        if bits[self.n_qubits - 1 - q_idx] == '0':
                            p0 += count
                        else:
                            p1 += count
                    return (p0 - p1) / self.shots
                
                ex = get_exp(counts_x, q)
                ey = get_exp(counts_y, q)
                ez = get_exp(counts_z, q)
                bloch_vectors[i, 3*q : 3*q+3] = [ex, ey, ez]
                
        return bloch_vectors

    def _compute_rbf_kernel(self, X_bloch: np.ndarray, Y_bloch: np.ndarray) -> np.ndarray:
        n, m = len(X_bloch), len(Y_bloch)
        K = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                diff = X_bloch[i] - Y_bloch[j]
                K[i, j] = np.exp(-self.gamma * np.dot(diff, diff))
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