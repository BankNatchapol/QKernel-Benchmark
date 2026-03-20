from __future__ import annotations

from typing import Union, Sequence, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

try:
    from feature_maps.base import FeatureMap
except ImportError:
    from .base import FeatureMap


def _get_x_params(num_qubits: int, x: Optional[Sequence] = None):
    """
    Return a feature vector of length num_qubits.
    If x is None, create symbolic parameters x[0], ..., x[n-1].
    """
    if x is None:
        return ParameterVector("x", num_qubits)
    if len(x) != num_qubits:
        raise ValueError(f"x must have length {num_qubits}, got {len(x)}")
    return x


def _pair_feature_angle(x, i: int, j: int, alpha: float = 1.0):
    """
    Pairwise interaction angle for controlled rotations.
    Default choice: alpha * x_i * x_j
    """
    return alpha * x[i] * x[j]


def _apply_reverse_chain_entanglement(
    qc: QuantumCircuit,
    x,
    gate: str = "crx",
    alpha: float = 1.0,
    circular: bool = False,
):
    """
    Apply the nearest-neighbor reverse-direction chain suggested by the sketch:
        q_n -> q_{n-1}, q_{n-1} -> q_{n-2}, ..., q_2 -> q_1
    """
    n = qc.num_qubits

    for i in range(n - 1, 0, -1):
        angle = _pair_feature_angle(x, i, i - 1, alpha)
        if gate == "crx":
            qc.crx(angle, i, i - 1)
        elif gate == "crz":
            qc.crz(angle, i, i - 1)
        elif gate == "cx":
            qc.cx(i, i - 1)
        else:
            raise ValueError(f"Unsupported entangling gate: {gate}")

    if circular and n > 2:
        angle = _pair_feature_angle(x, 0, n - 1, alpha)
        if gate == "crx":
            qc.crx(angle, 0, n - 1)
        elif gate == "crz":
            qc.crz(angle, 0, n - 1)
        elif gate == "cx":
            qc.cx(0, n - 1)


class CustomFeatureMap(FeatureMap):
    """
    A Generalized implementation of the custom feature maps described in the user request.
    """
    def __init__(
        self, 
        n_qubits: int, 
        reps: int = 1, 
        single_qubit: str = "ry", 
        entangler: str = "crx", 
        alpha: float = 1.0, 
        circular: bool = False
    ):
        super().__init__(n_qubits, reps)
        self.single_qubit = single_qubit
        self.entangler = entangler
        self.alpha = alpha
        self.circular = circular

    def build(self, x: np.ndarray | ParameterVector | None = None) -> QuantumCircuit:
        x_params = _get_x_params(self.n_qubits, x)
        qc = QuantumCircuit(self.n_qubits)

        for _ in range(self.reps):
            for i in range(self.n_qubits):
                if self.single_qubit == "rx_rz":
                    qc.rx(x_params[i], i)
                    qc.rz(x_params[i], i)
                elif self.single_qubit == "rz_rx":
                    qc.rz(x_params[i], i)
                    qc.rx(x_params[i], i)
                elif self.single_qubit == "ry":
                    qc.ry(x_params[i], i)
                elif self.single_qubit == "rx":
                    qc.rx(x_params[i], i)
                elif self.single_qubit == "rz":
                    qc.rz(x_params[i], i)
                else:
                    raise ValueError(f"Unknown single_qubit pattern: {self.single_qubit}")

            _apply_reverse_chain_entanglement(
                qc, x_params, gate=self.entangler, alpha=self.alpha, circular=self.circular
            )

        return qc


# -------------------------------------------------------------------
# Pre-defined Custom Feature Maps (CFM-1 through CFM-4)
# -------------------------------------------------------------------

class CFM1(CustomFeatureMap):
    """Feature Map 1: Rx + Rz + CRx chain."""
    def __init__(self, n_qubits: int, reps: int = 1, alpha: float = 1.0, circular: bool = False):
        super().__init__(n_qubits, reps, single_qubit="rx_rz", entangler="crx", alpha=alpha, circular=circular)

class CFM2(CustomFeatureMap):
    """Feature Map 2: Layered Rz + Rx + CRx chain."""
    def __init__(self, n_qubits: int, reps: int = 2, alpha: float = 1.0, circular: bool = False):
        super().__init__(n_qubits, reps, single_qubit="rz_rx", entangler="crx", alpha=alpha, circular=circular)

class CFM3(CustomFeatureMap):
    """Feature Map 3: Ry + CRz chain."""
    def __init__(self, n_qubits: int, reps: int = 1, alpha: float = 1.0, circular: bool = False):
        super().__init__(n_qubits, reps, single_qubit="ry", entangler="crz", alpha=alpha, circular=circular)

class CFM4(CustomFeatureMap):
    """Feature Map 4: Ry + CRx chain."""
    def __init__(self, n_qubits: int, reps: int = 1, alpha: float = 1.0, circular: bool = False):
        super().__init__(n_qubits, reps, single_qubit="ry", entangler="crx", alpha=alpha, circular=circular)
