"""
Checked by Bank
2026-03-07
"""

"""
ZZ-Feature Map (Havlíček et al., 2019).

Uses Qiskit's built-in ZZFeatureMap which encodes data into the
quantum state via parameterized ZZ interactions.

Reference:
    Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. Nature, 567, 209–212.
    https://doi.org/10.1038/s41586-019-0980-2

    Qiskit implementation:
    https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.circuit.library.ZZFeatureMap.html
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map

import sys
from pathlib import Path

# Allow direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .base import FeatureMap
except ImportError:
    from feature_maps.base import FeatureMap


class ZZMap(FeatureMap):
    """Wraps Qiskit's zz_feature_map, binding data x at call time."""

    def __init__(self, n_qubits: int, reps: int = 2):
        super().__init__(n_qubits, reps)
        self._template = zz_feature_map(feature_dimension=n_qubits, reps=reps)

    @FeatureMap.n_qubits.setter
    def n_qubits(self, value: int):
        self._n_qubits = value
        self._template = zz_feature_map(feature_dimension=value, reps=self._reps)

    def build(self, x: np.ndarray) -> QuantumCircuit:
        params = self._template.parameters
        values = [x[i % len(x)] for i in range(len(params))]
        bound = self._template.assign_parameters(dict(zip(params, values)))
        return bound

if __name__ == "__main__":
    zz_map = ZZMap(n_qubits=2, reps=2)
    x = np.array([0.1, 0.2])
    qc = zz_map.build(x)
    print(qc.draw())