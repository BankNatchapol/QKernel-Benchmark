from __future__ import annotations

import time
from typing import Any, Iterable
import numpy as np
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as IBM_Sampler

# Qiskit Machine Learning & Algorithms
try:
    from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
    from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_algorithms.optimizers import COBYLA, SPSA
except ImportError:
    pass

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


class _TranspilingSampler:
    """A wrapper for SamplerV2 that transpiles circuits to ISA just before running."""
    def __init__(self, sampler: Any, backend: Any):
        self._sampler = sampler
        self._backend = backend

    def run(self, pubs: Iterable, **options) -> Any:
        isa_pubs = []
        for pub in pubs:
            if isinstance(pub, tuple):
                circuit, values = pub[0], pub[1]
                t_qc = transpile(circuit, backend=self._backend, optimization_level=1)
                isa_pubs.append((t_qc, values))
            else:
                t_qc = transpile(pub, backend=self._backend, optimization_level=1)
                isa_pubs.append(t_qc)
        
        job = self._sampler.run(isa_pubs, **options)
        print(f"  QKTA: Job submitted (ID: {job.job_id()}). Waiting for results...")
        return job


def _build_trainable_circuit(
    n_qubits: int, reps: int = 1
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build a parameterized feature-map circuit U(x; theta)."""
    # Using distinct names to avoid name conflicts in Qiskit-ML
    x = ParameterVector("x_feat", n_qubits)
    theta = ParameterVector("theta_ker", n_qubits * reps)
    
    qc = QuantumCircuit(n_qubits)
    n_params_per_rep = n_qubits

    for r in range(reps):
        for i in range(n_qubits):
            angle = x[i] + theta[r * n_params_per_rep + i]
            qc.ry(angle, i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc, x, theta


class TrainableKernel(QuantumKernel):
    """Custom trainable fidelity quantum kernel optimized via KTA."""

    def __init__(
        self,
        n_qubits: int,
        reps: int = 1,
        shots: int = 1024,
        seed: int = 42,
        max_iter: int = 50,
        enforce_psd: bool = True,
        chunk_size: int = 4096,
        backend_name: str = "aer",
        backend: Any | None = None,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name, backend=backend)
        self.reps = reps
        self.max_iter = max_iter
        self.enforce_psd = enforce_psd
        
        # Build parameterized circuit
        self._qc, self._x_vec, self._theta_vec = _build_trainable_circuit(n_qubits, reps)
        # We store theta as a numpy array for consistency
        self._theta = np.zeros(len(self._theta_vec), dtype=float)
        
        # Backend setup
        if backend is not None:
            self._backend = backend
        elif backend_name == "statevector":
            self._backend = AerSimulator(method="statevector", seed_simulator=seed)
        else:
            self._backend = AerSimulator(seed_simulator=seed)

    @QuantumKernel.n_qubits.setter
    def n_qubits(self, value: int):
        self._n_qubits = value
        self._qc, self._x_vec, self._theta_vec = _build_trainable_circuit(value, self.reps)
        self._theta = np.zeros(len(self._theta_vec), dtype=float)

    def _validate_inputs(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2: raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_qubits: raise ValueError(f"X must have {self.n_qubits} features, got {X.shape[1]}")
        if Y is None: return X, None
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2: raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if Y.shape[1] != self.n_qubits: raise ValueError(f"Y must have {self.n_qubits} features, got {Y.shape[1]}")
        return X, Y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainableKernel":
        X, _ = self._validate_inputs(X)
        y = np.asarray(y)

        # 1. Setup Optimizers and Samplers
        if self.backend_name == "ibm":
            optimizer = SPSA(maxiter=self.max_iter)
            print(f"  QKTA: Using SPSA for hardware (maxiter={self.max_iter})")
            
            from qiskit_ibm_runtime import SamplerV2, Session
            try:
                print(f"  QKTA: Opening Session on {self._backend.name} for training iterations...")
                session = Session(backend=self._backend)
                # Wrap the sampler to handle ISA transpilation for ComputeUncompute's composed circuits
                t_sampler = _TranspilingSampler(SamplerV2(mode=session), self._backend)
            except Exception as e:
                # Fallback for Open Plan or other session limitations
                if "open plan" in str(e).lower() or "1352" in str(e):
                    print(f"  QKTA: Session not supported. Falling back to individual jobs.")
                    session = None
                    t_sampler = _TranspilingSampler(SamplerV2(mode=self._backend), self._backend)
                else:
                    raise e
        else:
            optimizer = COBYLA(maxiter=self.max_iter)
            print(f"  QKTA: Using COBYLA (maxiter={self.max_iter})")
            from qiskit.primitives import StatevectorSampler
            t_sampler = StatevectorSampler()
            session = None

        from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
        from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        
        try:
            t_fidelity = ComputeUncompute(sampler=t_sampler)
            
            t_kernel = TrainableFidelityQuantumKernel(
                feature_map=self._qc,
                fidelity=t_fidelity,
                training_parameters=self._theta_vec,
            )

            trainer = QuantumKernelTrainer(
                quantum_kernel=t_kernel,
                optimizer=optimizer,
                initial_point=self._theta,
            )

            qkt_results = trainer.fit(X, y)
        finally:
            if session:
                session.close()
                print("  QKTA: Session closed.")

        opt_params = qkt_results.optimal_parameters
        if isinstance(opt_params, dict):
            self._theta = np.array([opt_params[p] for p in self._theta_vec])
        else:
            self._theta = np.array(opt_params)
        
        print(f"  QKTA fit complete. Optimal KTA score: {qkt_results.optimal_value:.4f}")
        return self

    def build_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        self._reset_stats()
        t0 = time.perf_counter()

        X, Y = self._validate_inputs(X, Y)
        symmetric = Y is None
        Y = X if Y is None else Y
        n, m = len(X), len(Y)

        # 1. Bind learned theta once
        theta_dict = {self._theta_vec[i]: self._theta[i] for i in range(len(self._theta_vec))}
        qc_theta = self._qc.assign_parameters(theta_dict)

        if self.backend_name == "statevector":
            K = self._build_kernel_matrix_sv(X, Y, symmetric, n, m, qc_theta)
        elif self.backend_name == "ibm":
            K = self._build_kernel_matrix_ibm(X, Y, symmetric, n, m, qc_theta)
        else:
            K = self._build_kernel_matrix_aer(X, Y, symmetric, n, m, qc_theta)

        if symmetric and self.enforce_psd:
            K = self._project_to_psd(K)
            np.fill_diagonal(K, 1.0)

        self.stats.wall_clock_seconds = time.perf_counter() - t0
        return K

    def _build_kernel_matrix_sv(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool, n: int, m: int, qc_theta: QuantumCircuit
    ) -> np.ndarray:
        from qiskit.quantum_info import Statevector
        self.stats.update_stats(analyze_circuit_resources(qc_theta))

        total_pairs = (n * (n + 1)) // 2 if symmetric else n * m
        self.stats.total_shots = total_pairs * self.shots
        self.stats.n_evaluations = total_pairs

        def encode(data: np.ndarray, desc: str) -> np.ndarray:
            svs = []
            for i in tqdm(range(len(data)), desc=desc, unit="sv", ncols=88, leave=False):
                x_dict = {self._x_vec[k]: data[i, k] for k in range(self.n_qubits)}
                bound_x = qc_theta.assign_parameters(x_dict)
                svs.append(Statevector(bound_x).data)
            return np.array(svs)

        V_x = encode(X, "  QKTA SV (X)")
        V_y = V_x if symmetric else encode(Y, "  QKTA SV (Y)")
        K_exact = np.abs(V_x.conj() @ V_y.T)**2
        K_exact = np.clip(K_exact, 0.0, 1.0)

        rng = np.random.default_rng(self.seed)
        K = rng.binomial(self.shots, K_exact) / self.shots
        if symmetric: np.fill_diagonal(K, 1.0)
        return K

    def _build_kernel_matrix_aer(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool, n: int, m: int, qc_theta: QuantumCircuit
    ) -> np.ndarray:
        K = np.zeros((n, m), dtype=float)
        total_pairs = (n * (n + 1)) // 2 if symmetric else n * m
        chunk_size = self.chunk_size
        circuits, indices = [], []
        pbar = tqdm(total=total_pairs, desc="  QKTA matrix", unit="pair", ncols=88, leave=False)

        def _run_chunk() -> None:
            if not circuits: return
            if self.stats.total_shots == 0:
                self.stats.update_stats(analyze_circuit_resources(circuits[0]))
            
            self.stats.total_shots += self.shots * len(circuits)
            self.stats.n_evaluations += len(circuits)

            t_circs = transpile(circuits, self._backend, optimization_level=0)
            job = self._backend.run(t_circs, shots=self.shots)
            counts_list = job.result().get_counts()
            if not isinstance(counts_list, list): counts_list = [counts_list]

            zero_key = "0" * self.n_qubits
            for count, (idx_i, idx_j) in zip(counts_list, indices):
                val = count.get(zero_key, 0) / self.shots
                K[idx_i, idx_j] = val
                if symmetric and idx_i != idx_j: K[idx_j, idx_i] = val
            pbar.update(len(circuits))
            circuits.clear(); indices.clear()

        for i in range(n):
            start_j = i if symmetric else 0
            for j in range(start_j, m):
                if symmetric and i == j:
                    K[i, i] = 1.0; pbar.update(1); continue
                
                x_dict = {self._x_vec[k]: X[i, k] for k in range(self.n_qubits)}
                y_dict = {self._x_vec[k]: Y[j, k] for k in range(self.n_qubits)}
                
                phi_x = qc_theta.assign_parameters(x_dict)
                phi_y = qc_theta.assign_parameters(y_dict)
                
                bound_overlap = QuantumCircuit(self.n_qubits)
                bound_overlap.compose(phi_x, inplace=True)
                bound_overlap.compose(phi_y.inverse(), inplace=True)
                bound_overlap.measure_all()
                
                circuits.append(bound_overlap); indices.append((i, j))
                if len(circuits) >= chunk_size: _run_chunk()
        
        if circuits: _run_chunk()
        pbar.close()
        return K

    def _build_kernel_matrix_ibm(
        self, X: np.ndarray, Y: np.ndarray, symmetric: bool, n: int, m: int, qc_theta: QuantumCircuit
    ) -> np.ndarray:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        K = np.zeros((n, m), dtype=float)
        total_pairs = (n * (n + 1)) // 2 if symmetric else n * m
        
        xp = ParameterVector("xp", self.n_qubits)
        yp = ParameterVector("yp", self.n_qubits)
        
        phi_xp = qc_theta.assign_parameters({self._x_vec[k]: xp[k] for k in range(self.n_qubits)})
        phi_yp = qc_theta.assign_parameters({self._x_vec[k]: yp[k] for k in range(self.n_qubits)})
        
        template = QuantumCircuit(self.n_qubits)
        template.compose(phi_xp, inplace=True)
        template.compose(phi_yp.inverse(), inplace=True)
        template.measure_all()

        print(f"  QKTA IBM: Transpiling template for {self._backend.name}...")
        t_template = transpile(template, self._backend, optimization_level=3)
        self.stats.update_stats(analyze_circuit_resources(t_template))

        sampler = Sampler(mode=self._backend)
        pbar = tqdm(total=total_pairs, desc="  QKTA IBM", unit="pair", ncols=88, leave=False)
        circuits, indices = [], []

        def _run_chunk() -> None:
            if not circuits: return
            self.stats.total_shots += self.shots * len(circuits)
            self.stats.n_evaluations += len(circuits)
            job = sampler.run(circuits, shots=self.shots)
            print(f"\n  QKTA IBM: Job submitted (ID: {job.job_id()}). Waiting for results...")
            res = job.result()
            zero_key = "0" * self.n_qubits
            for pub_idx, (idx_i, idx_j) in enumerate(indices):
                counts = res[pub_idx].data.meas.get_counts()
                val = counts.get(zero_key, 0) / self.shots
                K[idx_i, idx_j] = val
                if symmetric and idx_i != idx_j: K[idx_j, idx_i] = val
            pbar.update(len(circuits))
            circuits.clear(); indices.clear()

        for i in range(n):
            start_j = i if symmetric else 0
            for j in range(start_j, m):
                if symmetric and i == j:
                    K[i, i] = 1.0; pbar.update(1); continue
                
                bind_dict = {}
                for k in range(self.n_qubits):
                    bind_dict[xp[k]] = X[i, k]
                    bind_dict[yp[k]] = Y[j, k]
                
                bound_qc = t_template.assign_parameters(bind_dict)
                circuits.append(bound_qc); indices.append((i, j))
                if len(circuits) >= self.chunk_size: _run_chunk()
        
        if circuits: _run_chunk()
        pbar.close()
        return K


if __name__ == "__main__":
    from datasets.loader import load_dataset
    from classifiers.qsvm import QSVM
    from sklearn.metrics import accuracy_score

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset("circles", n_samples=40, n_features=2)
    
    # Initialize and train
    kernel = TrainableKernel(n_qubits=2, reps=1, max_iter=10, shots=1024)
    print("\nTraining kernel...")
    kernel.fit(X_train, y_train)
    
    # Evaluate
    K_train = kernel.build_kernel_matrix(X_train)
    K_test = kernel.build_kernel_matrix(X_test, X_train)
    
    model = QSVM(C=1.0)
    model.fit(K_train, y_train)
    y_pred = model.predict(K_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Resource Stats:\n{kernel.stats}")