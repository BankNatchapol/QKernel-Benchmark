from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
import numpy as np

try:
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    print("Successfully initialized ComputeUncompute with StatevectorSampler")
except Exception as e:
    print(f"Failed to initialize ComputeUncompute with StatevectorSampler: {e}")

from qiskit_ibm_runtime import SamplerV2
try:
    # Just a mock check
    fidelity = ComputeUncompute(sampler=SamplerV2)
    print("Successfully checked ComputeUncompute with SamplerV2 class")
except Exception as e:
    print(f"Failed to check ComputeUncompute with SamplerV2: {e}")
