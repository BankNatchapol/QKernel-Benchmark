import os
import argparse
import sys
from pathlib import Path
import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Import Fake Providers
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeSherbrooke, FakeKyiv, FakeOsaka
    FAKE_BACKENDS = {
        "torino": FakeTorino,
        "sherbrooke": FakeSherbrooke,
        "kyiv": FakeKyiv,
        "osaka": FakeOsaka,
    }
except ImportError:
    FAKE_BACKENDS = {}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel
from benchmark.runner import BenchmarkRunner


def scale_probability_vector(probs, factor, cap=0.999):
    """Scale a probability vector (index 0 is identity)."""
    if len(probs) <= 1:
        return probs
    perr = sum(probs[1:])
    if perr <= 0:
        return probs

    new_perr = min(perr * factor, cap)
    scale = new_perr / perr

    new_probs = [0.0] * len(probs)
    new_probs[0] = 1.0 - new_perr
    for k in range(1, len(probs)):
        new_probs[k] = probs[k] * scale
    return new_probs


def scale_readout_row(row, factor, cap=0.999):
    """Scale a readout confusion matrix row."""
    n = len(row)
    # The diagonal element (max value) is the "correct" readout
    diag_idx = max(range(n), key=lambda j: row[j])

    offdiag_sum = sum(row[j] for j in range(n) if j != diag_idx)
    if offdiag_sum <= 0:
        return row[:]

    new_offdiag_sum = min(offdiag_sum * factor, cap)
    scale = new_offdiag_sum / offdiag_sum

    new_row = [0.0] * n
    for j in range(n):
        if j != diag_idx:
            new_row[j] = row[j] * scale
    new_row[diag_idx] = 1.0 - sum(new_row)
    return new_row


def adjust_noise_model(noise_model: NoiseModel, factor: float) -> NoiseModel:
    """
    Scale the probabilities in the noise model by a given factor.
    """
    if factor == 1.0:
        return noise_model
    
    nm_dict = noise_model.to_dict()
    
    if 'errors' in nm_dict:
        for error in nm_dict['errors']:
            if 'probabilities' in error:
                raw_probs = error['probabilities']
                
                if isinstance(raw_probs[0], list): # Matrix (ReadoutError)
                    new_matrix = []
                    for row in raw_probs:
                        new_matrix.append(scale_readout_row(row, factor))
                    error['probabilities'] = new_matrix
                else: # Vector (GateError)
                    error['probabilities'] = scale_probability_vector(raw_probs, factor)

    # Sanity checks
    for error in nm_dict.get('errors', []):
        probs = error.get('probabilities', [])
        if not probs: continue
        if isinstance(probs[0], list): # Matrix
            for row in probs:
                if not np.isclose(sum(row), 1.0, atol=1e-5):
                    raise ValueError(f"Readout row sum {sum(row)} != 1.0 after scaling")
                if any(p < 0 for p in row):
                    raise ValueError(f"Negative probability found in readout row after scaling")
        else: # Vector
            if not np.isclose(sum(probs), 1.0, atol=1e-5):
                raise ValueError(f"Probability vector sum {sum(probs)} != 1.0 after scaling")
            if any(p < 0 for p in probs):
                raise ValueError(f"Negative probability found in vector after scaling")

    new_nm = NoiseModel.from_dict(nm_dict)
    print(f"📊 Noise model adjusted by factor {factor} (regime: {'low' if factor < 1 else 'high' if factor > 1 else 'realistic'}).")
    return new_nm


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on a noisy simulator (Fake Backend).")
    parser.add_argument("--kernel", type=str, default="fidelity", choices=["fidelity", "projected", "trainable", "qflair"], help="Kernel type.")
    parser.add_argument("--dataset", type=str, default="circles", help="Dataset name.")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits/features.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples.")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit.")
    parser.add_argument("--device", type=str, default="torino", choices=list(FAKE_BACKENDS.keys()), help="IBM device to mimic.")
    parser.add_argument("--noise_factor", type=float, default=1.0, help="Scale factor for noise (1.0 = realistic).")
    parser.add_argument("--chunk", type=int, default=100, help="Circuits per job (chunk size).")
    
    args = parser.parse_args()

    # 1. Setup Fake Backend and Noise
    print(f"🛠️ Setting up noisy simulation for mimicking: {args.device}...")
    backend_class = FAKE_BACKENDS.get(args.device)
    if not backend_class:
        print(f"❌ Error: Fake backend for {args.device} not found.")
        return
    
    real_backend = backend_class()
    noise_model = NoiseModel.from_backend(real_backend)
    
    if args.noise_factor != 1.0:
        noise_model = adjust_noise_model(noise_model, args.noise_factor)

    # AerSimulator with noise model and coupling map
    sim_backend = AerSimulator.from_backend(real_backend)
    
    print(f"✅ Simulator initialized with noise model from {args.device}.")

    # 2. Initialize Kernel
    # We use the 'ibm' path in kernels because it uses SamplerV2 and transpilation,
    # which is exactly what we want for a realistic hardware-like simulation.
    print(f"Initializing {args.kernel} kernel...")
    
    kernel_kwargs = {
        "n_qubits": args.qubits,
        "shots": args.shots,
        "backend_name": "ibm", # Use IBM path for realistic transpilation/primitives
        "backend": sim_backend,
        "chunk_size": args.chunk
    }

    if args.kernel == "fidelity":
        kernel = FidelityKernel(**kernel_kwargs)
    elif args.kernel == "projected":
        kernel = ProjectedKernel(**kernel_kwargs)
    elif args.kernel == "trainable":
        kernel = TrainableKernel(**kernel_kwargs, max_iter=20)
    elif args.kernel == "qflair":
        kernel = QFLAIRKernel(**kernel_kwargs, n_layers=2)

    # 3. Run Experiment
    runner = BenchmarkRunner(
        kernels={args.kernel: kernel},
        dataset_names=[args.dataset],
        n_qubits=args.qubits,
        shots=args.shots,
        n_samples=args.samples,
        results_dir="results_noisy"
    )

    print(f"\n🚀 Starting Noisy Simulation: {args.kernel} on {args.dataset} ({args.samples} samples) mimicking {args.device}")
    results = runner.run_one(args.kernel, kernel, args.dataset)
    
    print("\n=== Simulation Results ===")
    for k, v in results.items():
        if k not in ["confusion_matrix", "roc_curve"]:
            print(f"{k}: {v}")

    print(f"\n✅ Results saved to results_noisy/")

if __name__ == "__main__":
    main()
