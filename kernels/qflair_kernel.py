"""
Checked by Bank
2026-03-09
"""

from __future__ import annotations

"""
Q-FLAIR-inspired QSVM fidelity kernel with greedy feature-map learning.

This module implements a simplified, paper-aligned variant of the QSVM branch of
Q-FLAIR ("Quantum feature-map learning with reduced resource overhead"),
together with the kernel-target-alignment (KTA) viewpoint from
"Training Quantum Embedding Kernels on Near-Term Quantum Computers".

================================================================================
THEORY / PAPER CONTEXT
================================================================================

1) Trainable quantum embedding kernels
--------------------------------------
A quantum kernel method starts from a data-dependent feature map U(x), which
embeds a classical sample x into a quantum state

    |phi(x)> = U(x) |0...0>.

The fidelity kernel between two samples x and x' is then

    k(x, x') = | <phi(x') | phi(x)> |^2.

This is the core kernel viewpoint in:
    - Hubregtsen et al., "Training Quantum Embedding Kernels on Near-Term
      Quantum Computers", arXiv:2105.02276
      https://arxiv.org/abs/2105.02276

That paper also motivates optimizing the kernel with label information using
kernel-target alignment (KTA), so that samples from the same class tend to
be more similar than samples from different classes.

2) Q-FLAIR
----------
Q-FLAIR ("Quantum feature-map learning with reduced resource overhead",
arXiv:2510.03389) proposes a more resource-frugal way to learn a quantum
feature map. Instead of training a fixed ansatz with repeated full quantum
evaluations, Q-FLAIR greedily grows the circuit one gate at a time.

For each candidate appended gate, it uses a small number of quantum evaluations
to reconstruct a simple sinusoidal dependence on the new gate's parameter, and
then performs the feature / weight search classically.

Reference:
    - Haas et al., "Quantum feature-map learning with reduced resource overhead",
      arXiv:2510.03389
      https://arxiv.org/abs/2510.03389

================================================================================
WHAT THIS IMPLEMENTATION MATCHES
================================================================================

This implementation is intentionally a *simplified QSVM-oriented variant*.

It matches the main Q-FLAIR QSVM structure:

    - greedy feature-map growth from an empty circuit
    - candidate pool of appended data-dependent gates
    - three-point reconstruction of a cosine dependence
    - classical search over feature index and weight
    - final fidelity kernel built from the learned feature map
    - KTA-based scoring

It is therefore aligned with:
    (a) the general trainable-kernel / KTA theory of arXiv:2105.02276
    (b) the algorithmic spirit of the QSVM branch of Q-FLAIR

================================================================================
IMPORTANT DIFFERENCES VS THE ORIGINAL Q-FLAIR REPOSITORY
================================================================================

This is NOT a byte-for-byte reproduction of the original repository. Important
simplifications / design choices are:

1) Reduced gate pool
   The original Q-FLAIR repository exposes a much broader symbolic gate pool.
   Here we use only a practical subset:

       { Rz, Rxx, Ryy, Rzz }

   over selected wires.

2) Cleaner object-oriented API
   The original repository is script-style and parameter-file driven.
   This file provides a reusable class-based interface.

3) Exact kernel refresh after accepting a gate
   After each greedy step, this implementation recomputes the *exact sampled*
   kernel matrix of the newly learned circuit and updates the current KTA from
   that exact kernel.
   This was added deliberately to keep the greedy sequence better anchored to
   the true learned model.

4) Optional clipping of reconstructed kernels
   Reconstructed kernels are not clipped by default. For an exact fidelity
   kernel, entries lie in [0, 1], but a reconstructed surrogate may overshoot.
   Clipping is therefore exposed only as an optional engineering choice.

5) Scaling convention
   The original QSVM script scales features with MinMaxScaler(feature_range=(0, pi)).
   To be close to that implementation, inputs to this class are expected to
   already be scaled to [0, pi].

================================================================================
HIGH-LEVEL ALGORITHM
================================================================================

Given training data X and labels y:

1) Start with an empty learned circuit U_0(x).
2) For each layer / greedy iteration:
   a) For each gate candidate g:
      i)   Probe the candidate with three relative angles:
               alpha0, alpha0 + pi/2, alpha0 - pi/2
      ii)  Reconstruct pairwise kernel entries
               k_ij(alpha) = a_ij cos(alpha - b_ij) + c_ij
      iii) For each feature component k:
               set alpha_ij = w * (x_i,k - x_j,k)
               optimize w classically to maximize KTA
   b) Choose the best (candidate gate, feature, weight)
   c) Append that gate to the learned circuit
   d) Recompute the exact sampled kernel of the updated learned circuit
      and update the current exact KTA
   e) Stop if the gain is below a threshold
3) Use the final learned circuit as a fidelity kernel for QSVM or other
   kernel-based downstream methods.

================================================================================
PRACTICAL NOTES
================================================================================

- Inputs should be preprocessed / scaled before fitting. To emulate the
  original QSVM repository most closely, scale features to [0, pi].
- This implementation uses overlap circuits with measurements to estimate
  fidelity entries by the all-zero probability.
- The reconstruction step is a surrogate used for candidate selection;
  the accepted greedy state is updated using the exact sampled kernel.
"""

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
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


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class GateCandidate:
    """A candidate gate type to append during greedy Q-FLAIR."""
    name: str
    wires: tuple[int, ...]


@dataclass(frozen=True)
class LearnedGate:
    """A learned data-dependent gate in the final feature map."""
    name: str
    wires: tuple[int, ...]
    feature_idx: int
    weight: float


# =============================================================================
# Gate utilities
# =============================================================================

_ALLOWED_GATES = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}


def _validate_gate_candidate(candidate: GateCandidate) -> None:
    if candidate.name.lower() not in _ALLOWED_GATES:
        raise ValueError(
            f"Unsupported gate candidate '{candidate.name}'. "
            f"Allowed: {sorted(_ALLOWED_GATES)}"
        )


def _apply_weight_data_gate(
    qc: QuantumCircuit,
    gate: LearnedGate | GateCandidate,
    x: np.ndarray,
    alpha_override: float | None = None,
) -> None:
    """
    Apply a data-dependent gate to the circuit.

    Parameters
    ----------
    qc:
        Circuit to modify.
    gate:
        Either a learned gate (feature index + weight already chosen) or a
        candidate gate.
    x:
        Input sample.
    alpha_override:
        If provided, use this angle directly.
        If None, require a LearnedGate and compute

            alpha = weight * x[feature_idx].

    Notes
    -----
    In the greedy reconstruction phase, we often want to probe a candidate gate
    at explicit angles alpha. In that case, alpha_override is used and the
    feature dependence is substituted later at the classical level.
    """
    name = gate.name.lower()

    if alpha_override is None:
        if not isinstance(gate, LearnedGate):
            raise ValueError(
                "alpha_override must be provided when applying a GateCandidate."
            )
        alpha = gate.weight * float(x[gate.feature_idx])
        wires = gate.wires
    else:
        alpha = float(alpha_override)
        wires = gate.wires

    if name == "rx":
        (q,) = wires
        qc.rx(alpha, q)
    elif name == "ry":
        (q,) = wires
        qc.ry(alpha, q)
    elif name == "rz":
        (q,) = wires
        qc.rz(alpha, q)
    elif name == "rxx":
        q0, q1 = wires
        qc.rxx(alpha, q0, q1)
    elif name == "ryy":
        q0, q1 = wires
        qc.ryy(alpha, q0, q1)
    elif name == "rzz":
        q0, q1 = wires
        qc.rzz(alpha, q0, q1)
    else:
        raise ValueError(f"Unsupported gate '{gate.name}'")


# =============================================================================
# Q-FLAIR-inspired QSVM kernel
# =============================================================================

class QFLAIRKernel(QuantumKernel):
    """
    Q-FLAIR-inspired fidelity kernel with greedy feature-map learning.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the feature map.
    n_layers:
        Maximum number of greedy gate-addition steps.
    shots:
        Number of shots for sampled overlap circuits.
    seed:
        Random seed for the simulator.
    feature_weight_bounds:
        Bounds for the scalar feature weight w in alpha = w * (x_i - x_j).
    alpha0:
        Center angle used for the three-point reconstruction:
            alpha0, alpha0 + pi/2, alpha0 - pi/2
    candidate_pool:
        Optional custom gate pool. If None, a default practical subset is used.
    weight_opt_maxiter:
        Max iterations for the scalar optimizer over the gate weight.
    min_gain:
        Minimum exact-KTA improvement required to accept a new gate.
    clip_reconstructed_kernel:
        Whether to clip the reconstructed surrogate kernel into [0, 1].
        Default False for cleaner theory behavior.
    optimize_scalar_method:
        Method passed to scipy.optimize.minimize_scalar.
        "bounded" is the default practical choice.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 6,
        shots: int = 1024,
        seed: int = 42,
        feature_weight_bounds: tuple[float, float] = (-1.0, 1.0),
        alpha0: float = 0.0,
        candidate_pool: list[GateCandidate] | None = None,
        weight_opt_maxiter: int = 60,
        min_gain: float = 1e-4,
        clip_reconstructed_kernel: bool = False,
        optimize_scalar_method: str = "bounded",
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)

        self.n_layers = int(n_layers)
        self.alpha0 = float(alpha0)
        self.weight_opt_maxiter = int(weight_opt_maxiter)
        self.min_gain = float(min_gain)
        self.weight_bounds = tuple(float(v) for v in feature_weight_bounds)
        self.clip_reconstructed_kernel = bool(clip_reconstructed_kernel)
        self.optimize_scalar_method = str(optimize_scalar_method)

        self._backend = AerSimulator(seed_simulator=seed)
        self._learned_gates: list[LearnedGate] = []

        if candidate_pool is None:
            pool: list[GateCandidate] = []

            # Single-qubit rotations on every qubit
            for q in range(n_qubits):
                pool.append(GateCandidate("rx", (q,)))
                pool.append(GateCandidate("ry", (q,)))
                pool.append(GateCandidate("rz", (q,)))

            # Nearest-neighbor entangling two-qubit gates
            for q in range(n_qubits - 1):
                pool.append(GateCandidate("rxx", (q, q + 1)))
                pool.append(GateCandidate("ryy", (q, q + 1)))
                pool.append(GateCandidate("rzz", (q, q + 1)))

            candidate_pool = pool

        for cand in candidate_pool:
            _validate_gate_candidate(cand)

        self.candidate_pool = list(candidate_pool)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_inputs(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[1] < 1:
            raise ValueError("X must contain at least one feature")

        if Y is None:
            return X, None

        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if Y.shape[1] != X.shape[1]:
            raise ValueError(
                f"X and Y must have the same feature dimension, got "
                f"{X.shape[1]} and {Y.shape[1]}"
            )

        return X, Y

    def _validate_labels(self, y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != n_samples:
            raise ValueError("X and y must have the same number of samples")
        if y.shape[0] == 0:
            raise ValueError("y must not be empty")
        return y

    # -------------------------------------------------------------------------
    # Circuit builders
    # -------------------------------------------------------------------------

    def _build_feature_map(
        self,
        x: np.ndarray,
        learned_gates: list[LearnedGate] | None = None,
    ) -> QuantumCircuit:
        """
        Build the current learned feature map U(x).

        The circuit starts empty and is grown greedily gate by gate.
        """
        if learned_gates is None:
            learned_gates = self._learned_gates

        qc = QuantumCircuit(self.n_qubits)
        for gate in learned_gates:
            _apply_weight_data_gate(qc, gate, x)
        return qc

    def _build_augmented_state_circuit(
        self,
        x: np.ndarray,
        learned_gates: list[LearnedGate],
        candidate: GateCandidate | None = None,
        alpha_override: float | None = None,
    ) -> QuantumCircuit:
        """
        Build U_current(x), optionally followed by one appended candidate gate.

        During the reconstruction step, the appended gate is probed at explicit
        angles alpha rather than via alpha = weight * x_k.
        """
        qc = self._build_feature_map(x, learned_gates)
        if candidate is not None:
            _apply_weight_data_gate(
                qc,
                candidate,
                x,
                alpha_override=alpha_override,
            )
        return qc

    def _build_overlap_circuit(
        self,
        x: np.ndarray,
        x_prime: np.ndarray,
        learned_gates: list[LearnedGate] | None = None,
    ) -> QuantumCircuit:
        """
        Build the standard overlap circuit for the final learned map.

        This estimates:
            k(x, x') = |<phi(x') | phi(x)>|^2
        as the probability of measuring all zeros after
            U(x) U(x')^†
        is applied to |0...0>.
        """
        if learned_gates is None:
            learned_gates = self._learned_gates

        phi_x = self._build_feature_map(x, learned_gates)
        phi_xp = self._build_feature_map(x_prime, learned_gates)

        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.compose(phi_x, inplace=True)
        qc.compose(phi_xp.inverse(), inplace=True)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def _build_reconstruction_overlap_circuit(
        self,
        x: np.ndarray,
        x_prime: np.ndarray,
        learned_gates: list[LearnedGate],
        candidate: GateCandidate,
        alpha: float,
    ) -> QuantumCircuit:
        """
        Build the overlap circuit used during the reconstruction step.

        We append the candidate gate with:
            - explicit angle alpha on the x branch
            - explicit angle 0 on the x' branch

        This probes an effective relative-angle response. After reconstructing
        the cosine coefficients, we substitute the feature dependence classically
        through:
            alpha_ij = w * (x_i,k - x_j,k).

        This matches the intended QSVM-style cosine-in-difference structure used
        by the original Q-FLAIR QSVM implementation, while keeping the code
        simpler and modular.
        """
        phi_x = self._build_augmented_state_circuit(
            x,
            learned_gates,
            candidate=candidate,
            alpha_override=alpha,
        )
        phi_xp = self._build_augmented_state_circuit(
            x_prime,
            learned_gates,
            candidate=candidate,
            alpha_override=0.0,
        )

        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.compose(phi_x, inplace=True)
        qc.compose(phi_xp.inverse(), inplace=True)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    # -------------------------------------------------------------------------
    # Execution helpers
    # -------------------------------------------------------------------------

    def _run_overlap_batch(
        self,
        circuits: list[QuantumCircuit],
    ) -> list[float]:
        """
        Execute overlap circuits and return the all-zero probabilities.
        """
        if not circuits:
            return []

        t_circuits = transpile(
            circuits,
            self._backend,
            optimization_level=0,
            seed_transpiler=self.seed,
        )
        job = self._backend.run(t_circuits, shots=self.shots)
        counts_list = job.result().get_counts()

        if not isinstance(counts_list, list):
            counts_list = [counts_list]

        zero_key = "0" * self.n_qubits
        return [count.get(zero_key, 0) / self.shots for count in counts_list]

    # -------------------------------------------------------------------------
    # Kernel / KTA helpers
    # -------------------------------------------------------------------------

    def _build_kernel_matrix_for_sequence(
        self,
        X: np.ndarray,
        learned_gates: list[LearnedGate],
    ) -> np.ndarray:
        """
        Build the exact sampled fidelity kernel matrix for a fixed learned map.

        "Exact" here means exact with respect to the currently chosen circuit,
        but still estimated by shot-based overlap circuits.
        """
        n = len(X)
        K = np.eye(n, dtype=float)

        circuits: list[QuantumCircuit] = []
        indices: list[tuple[int, int]] = []

        for i in range(n):
            for j in range(i + 1, n):
                circuits.append(
                    self._build_overlap_circuit(X[i], X[j], learned_gates)
                )
                indices.append((i, j))

        vals = self._run_overlap_batch(circuits)
        for val, (i, j) in zip(vals, indices):
            K[i, j] = val
            K[j, i] = val

        return K

    @staticmethod
    def _kta(K: np.ndarray, y: np.ndarray) -> float:
        """
        Compute kernel-target alignment with target T = y y^T.

        KTA(K, y) = <K, yy^T>_F / (||K||_F ||yy^T||_F)

        This matches the label-aware kernel-training viewpoint of
        trainable quantum embedding kernels.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        T = np.outer(y, y)
        numerator = float(np.sum(K * T))
        denominator = float(
            np.sqrt(np.sum(K * K) * np.sum(T * T)) + 1e-12
        )
        return numerator / denominator

    @staticmethod
    def _reconstruct_cosine_coeffs(
        z0: np.ndarray,
        z_plus: np.ndarray,
        z_minus: np.ndarray,
        alpha0: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct coefficients a, b, c in

            f(alpha) = a cos(alpha - b) + c

        from the three probe values:
            z0     = f(alpha0)
            z_plus = f(alpha0 + pi/2)
            z_minus= f(alpha0 - pi/2)
        """
        s1 = 2.0 * z0 - z_plus - z_minus
        s2 = z_plus - z_minus

        # Mathematically exact inversion of the 3-point cosine probe:
        # z0 = a cos(alpha0 - b) + c
        # z_plus = -a sin(alpha0 - b) + c
        # z_minus = a sin(alpha0 - b) + c
        # => s1 = 2a cos(alpha0 - b)
        # => s2 = -2a sin(alpha0 - b)
        
        a = 0.5 * np.sqrt(s1 * s1 + s2 * s2)
        # alpha0 - b = arctan2(a sin(alpha0 - b), a cos(alpha0 - b)) = arctan2(-s2/2, s1/2)
        b = alpha0 - np.arctan2(-s2, s1)
        c = 0.5 * (z_plus + z_minus)

        return a, b, c

    def _candidate_reconstruction(
        self,
        X: np.ndarray,
        learned_gates: list[LearnedGate],
        candidate: GateCandidate,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For one candidate gate, reconstruct the pairwise kernel coefficients
        a_ij, b_ij, c_ij for all training pairs (i, j).

        The three probe angles are:
            alpha0
            alpha0 + pi/2
            alpha0 - pi/2
        """
        n = len(X)
        alpha0 = self.alpha0
        probe_alphas = [alpha0, alpha0 + np.pi / 2.0, alpha0 - np.pi / 2.0]

        value_maps: list[np.ndarray] = []

        for alpha in probe_alphas:
            circuits: list[QuantumCircuit] = []
            indices: list[tuple[int, int]] = []

            for i in range(n):
                for j in range(i + 1, n):
                    circuits.append(
                        self._build_reconstruction_overlap_circuit(
                            X[i], X[j], learned_gates, candidate, alpha
                        )
                    )
                    indices.append((i, j))

            vals = self._run_overlap_batch(circuits)

            M = np.eye(n, dtype=float)
            for val, (i, j) in zip(vals, indices):
                M[i, j] = val
                M[j, i] = val

            value_maps.append(M)

        a, b, c = self._reconstruct_cosine_coeffs(
            value_maps[0],
            value_maps[1],
            value_maps[2],
            alpha0,
        )

        # Enforce exact diagonal fidelity for the surrogate.
        np.fill_diagonal(a, 0.0)
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(c, 1.0)

        return a, b, c

    def _reconstructed_kernel_from_feature_weight(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        X: np.ndarray,
        feature_idx: int,
        weight: float,
    ) -> np.ndarray:
        """
        Build the reconstructed surrogate kernel matrix for a chosen feature
        component and scalar weight.

        The classically substituted relative angle is:

            alpha_ij = weight * (x_i,k - x_j,k)

        and the surrogate is:

            K_ij = a_ij cos(alpha_ij - b_ij) + c_ij
        """
        diff = X[:, feature_idx][:, None] - X[:, feature_idx][None, :]
        alpha = weight * diff

        K = a * np.cos(alpha - b) + c

        # Numerical symmetry cleanup.
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)

        if self.clip_reconstructed_kernel:
            K = np.clip(K, 0.0, 1.0)

        return K

    def _optimize_weight_for_feature(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int,
    ) -> tuple[float, float]:
        """
        Optimize the scalar weight for one chosen feature using the
        reconstructed surrogate KTA.

        Returns
        -------
        weight_star:
            Best scalar weight.
        kta_star:
            Best surrogate KTA at that weight.
        """
        def objective(weight: float) -> float:
            K_rec = self._reconstructed_kernel_from_feature_weight(
                a=a,
                b=b,
                c=c,
                X=X,
                feature_idx=feature_idx,
                weight=float(weight),
            )
            return -self._kta(K_rec, y)

        method = self.optimize_scalar_method.lower()
        if method == "bounded":
            result = minimize_scalar(
                objective,
                bounds=self.weight_bounds,
                method="bounded",
                options={"maxiter": self.weight_opt_maxiter},
            )
        else:
            # Fallback: still honor bounds through clipping inside the objective.
            result = minimize_scalar(
                objective,
                method=method,
                options={"maxiter": self.weight_opt_maxiter},
            )

        weight_star = float(result.x)
        # Ensure weight is kept inside the stated bounds
        low, high = self.weight_bounds
        weight_star = float(np.clip(weight_star, low, high))
        kta_star = -float(objective(weight_star))

        return weight_star, kta_star

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QFLAIRKernel":
        """
        Learn the feature map greedily.

        Crucial design choice:
        ----------------------
        Candidate scoring is done with the reconstructed surrogate kernel, but
        after a candidate is accepted we recompute the *exact sampled* kernel of
        the updated learned circuit and use that exact KTA as the new baseline.

        This keeps the greedy sequence better aligned with the true model than
        simply carrying forward the surrogate KTA.
        """
        X, _ = self._validate_inputs(X)
        y = self._validate_labels(y, len(X))

        learned: list[LearnedGate] = []

        # Exact baseline for the empty circuit:
        # U(x) = identity => all states are identical => K = all-ones matrix.
        current_K = np.ones((len(X), len(X)), dtype=float)
        current_kta = self._kta(current_K, y)

        n_features = X.shape[1]

        from tqdm import tqdm
        with tqdm(
            total=self.n_layers,
            desc="  QFLAIR fit",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ncols=88,
            leave=False,
        ) as pbar:
            for _layer in range(self.n_layers):
                best_choice: tuple[GateCandidate, int, float] | None = None
                best_surrogate_kta = -np.inf
    
                # Search candidate gate, feature, and weight.
                for candidate in self.candidate_pool:
                    a, b, c = self._candidate_reconstruction(X, learned, candidate)
    
                    for feature_idx in range(n_features):
                        weight_star, surrogate_kta = self._optimize_weight_for_feature(
                            a=a,
                            b=b,
                            c=c,
                            X=X,
                            y=y,
                            feature_idx=feature_idx,
                        )
    
                        if surrogate_kta > best_surrogate_kta:
                            best_surrogate_kta = surrogate_kta
                            best_choice = (candidate, feature_idx, weight_star)
    
                if best_choice is None:
                    break
    
                candidate, feature_idx, weight_star = best_choice
                proposed_gate = LearnedGate(
                    name=candidate.name,
                    wires=candidate.wires,
                    feature_idx=feature_idx,
                    weight=weight_star,
                )
    
                # IMPORTANT:
                # Refresh the greedy state using the exact sampled kernel after
                # actually appending the chosen gate.
                proposed_learned = [*learned, proposed_gate]
                exact_K_new = self._build_kernel_matrix_for_sequence(X, proposed_learned)
                exact_kta_new = self._kta(exact_K_new, y)
                exact_gain = exact_kta_new - current_kta
    
                if exact_gain < self.min_gain:
                    break
    
                learned = proposed_learned
                current_K = exact_K_new
                current_kta = exact_kta_new
                
                pbar.set_postfix({"kta": f"{current_kta:.4f}", "gates": len(learned)})
                pbar.update(1)

        self._learned_gates = learned
        return self

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build_kernel_matrix(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build the final sampled fidelity kernel matrix from the learned map.

        If Y is None, returns a symmetric n x n matrix.
        Otherwise returns an n x m cross-kernel matrix.
        """
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
                if symmetric and i == j:
                    K[i, i] = 1.0
                    continue

                circuits.append(
                    self._build_overlap_circuit(X[i], Y[j], self._learned_gates)
                )
                indices.append((i, j))

        if circuits:
            self.stats.total_shots = self.shots * len(circuits)
            self.stats.n_evaluations = len(circuits)

            # Resource stats from one representative overlap circuit
            res = analyze_circuit_resources(circuits[0])
            self.stats.total_depth = res["total_depth"]
            self.stats.two_qubit_depth = res["two_qubit_depth"]
            self.stats.total_gates = res["total_gates"]
            self.stats.two_qubit_count = res["two_qubit_count"]
            self.stats.one_qubit_count = res["one_qubit_count"]
            self.stats.gate_breakdown = res["gate_breakdown"]

            vals = self._run_overlap_batch(circuits)
            for val, (i, j) in zip(vals, indices):
                K[i, j] = val
                if symmetric:
                    K[j, i] = val
        else:
            self.stats.total_shots = 0
            self.stats.n_evaluations = 0

            if n > 0 and m > 0:
                example = self._build_overlap_circuit(
                    X[0], Y[0], self._learned_gates
                )
                res = analyze_circuit_resources(example)
                self.stats.total_depth = res["total_depth"]
                self.stats.two_qubit_depth = res["two_qubit_depth"]
                self.stats.total_gates = res["total_gates"]
                self.stats.two_qubit_count = res["two_qubit_count"]
                self.stats.one_qubit_count = res["one_qubit_count"]
                self.stats.gate_breakdown = res["gate_breakdown"]

        self.stats.wall_clock_seconds = time.perf_counter() - t0
        return K

    # -------------------------------------------------------------------------
    # Convenience accessors
    # -------------------------------------------------------------------------

    @property
    def learned_gates(self) -> list[LearnedGate]:
        """Return a copy of the learned gate sequence."""
        return list(self._learned_gates)

    def reset(self) -> None:
        """Clear the learned feature map."""
        self._learned_gates = []

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    from datasets.loader import load_dataset
    from kernels.qflair_kernel import QFLAIRKernel
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
    kernel = QFLAIRKernel(n_qubits=2, n_layers=2, shots=1024, seed=42)
    
    print("\nTraining kernel parameters (n_layers=2)...")
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

    # Final Architecture
    print("\nFinal Learned Architecture for X_train[0]:")
    final_qc = kernel._build_feature_map(X_train[0])
    print(final_qc.draw(fold=-1))