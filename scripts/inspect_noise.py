from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer.noise import NoiseModel
import numpy as np

backend = FakeTorino()
nm = NoiseModel.from_backend(backend)
nm_dict = nm.to_dict()

# Print the values in probabilities
if 'errors' in nm_dict:
    for error in nm_dict['errors']:
        if 'probabilities' in error:
            probs = error['probabilities']
            print(f"Probabilities type: {type(probs)}")
            if len(probs) > 0:
                print(f"First element type: {type(probs[0])}")
                print(f"First element value: {probs[0]}")
            break
