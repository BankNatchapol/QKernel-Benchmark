"""
Checked by Bank
2026-03-07
"""

"""Abstract base class for quantum feature maps."""

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit


class FeatureMap(ABC):
    """Abstract feature map: encodes a classical data vector into a quantum state."""

    def __init__(self, n_qubits: int, reps: int = 1):
        self._n_qubits = n_qubits
        self._reps = reps

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, value: int):
        self._n_qubits = value

    @property
    def reps(self) -> int:
        return self._reps

    @abstractmethod
    def build(self, x: np.ndarray) -> QuantumCircuit:
        """Return a QuantumCircuit encoding the data vector x."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self._n_qubits}, reps={self._reps})"
