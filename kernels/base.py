"""
Checked by Bank
2026-03-09
"""

"""Abstract base class for quantum kernels."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResourceStats:
    """Tracks quantum resource usage for a kernel computation."""
    n_qubits: int = 0
    total_shots: int = 0
    n_evaluations: int = 0
    wall_clock_seconds: float = 0.0
    
    # Detailed circuit resources (transpiled to common basis)
    total_depth: int = 0
    two_qubit_depth: int = 0
    total_gates: int = 0
    two_qubit_count: int = 0
    one_qubit_count: int = 0
    gate_breakdown: str = ""


class QuantumKernel(ABC):
    """Abstract quantum kernel.

    Subclasses implement ``build_kernel_matrix`` which accepts classical data
    arrays and returns a Gram matrix computed via quantum circuits.

    Resource usage is accumulated in ``self.stats`` during each call to
    ``build_kernel_matrix``.
    """

    def __init__(self, n_qubits: int, shots: int = 1024, seed: int = 42):
        self.n_qubits = n_qubits
        self.shots = shots
        self.seed = seed
        self.stats = ResourceStats(n_qubits=n_qubits)

    def _reset_stats(self) -> None:
        """Reset resource counters before a new kernel computation."""
        self.stats = ResourceStats(n_qubits=self.n_qubits)

    @abstractmethod
    def build_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute the (n×m) kernel matrix K where K[i,j] = k(X[i], Y[j]).

        If Y is None the kernel is evaluated as K[i,j] = k(X[i], X[j]) (square).
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_qubits={self.n_qubits}, shots={self.shots})"
        )
