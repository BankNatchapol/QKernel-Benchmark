import os
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

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

from experiments.run_single_noisy_sim import adjust_noise_model
from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel
from benchmark.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite on a noisy simulator across multiple regimes.")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits/features.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples per regime.")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit.")
    parser.add_argument("--device", type=str, default="torino", choices=list(FAKE_BACKENDS.keys()), help="IBM device to mimic.")
    parser.add_argument("--chunk", type=int, default=100, help="Circuits per job.")
    
    args = parser.parse_args()

    # Define noise regimes
    noise_regimes = {
        "low": 0.5,
        "realistic": 1.0,
        "high": 2.0
    }

    datasets = ["higgs", "hepmass", "energyflow"]
    final_dfs = []

    print(f"� Starting Multi-Regime Noisy Simulation Suite mimicking: {args.device}")
    
    for regime_name, factor in noise_regimes.items():
        print(f"\n--- 🛰️ Running Regime: {regime_name.upper()} (factor={factor}) ---")
        
        # 1. Setup Backend and Noise
        backend_class = FAKE_BACKENDS.get(args.device)
        real_backend = backend_class()
        
        noise_model = NoiseModel.from_backend(real_backend)
        if factor != 1.0:
            noise_model = adjust_noise_model(noise_model, factor)

        sim_backend = AerSimulator.from_backend(real_backend, noise_model=noise_model)
        
        # 2. Define Kernels for this backend
        kernel_kwargs = {
            "n_qubits": args.qubits,
            "shots": args.shots,
            "backend_name": "ibm",
            "backend": sim_backend,
            "chunk_size": args.chunk
        }

        kernels = {
            "FQK": FidelityKernel(**kernel_kwargs),
            "PQK": ProjectedKernel(**kernel_kwargs),
            "QKTA": TrainableKernel(**kernel_kwargs, max_iter=10),
        }

        # 3. Run Benchmark
        regime_results_dir = f"results_noisy_all/{regime_name}"
        runner = BenchmarkRunner(
            kernels=kernels,
            dataset_names=datasets,
            n_qubits=args.qubits,
            shots=args.shots,
            n_samples=args.samples,
            results_dir=regime_results_dir
        )

        results_df = runner.run()
        results_df["regime"] = regime_name
        final_dfs.append(results_df)

    # 4. Final Summary
    all_results_df = pd.concat(final_dfs, ignore_index=True)
    os.makedirs("results_noisy_all", exist_ok=True)
    all_results_df.to_csv("results_noisy_all/full_regime_summary.csv", index=False)
    
    print("\n" + "="*40)
    print("📋 MULTI-REGIME NOISY BENCHMARK COMPLETE")
    print("="*40)
    print(all_results_df.to_string())
    print("\n✅ All results, summaries, and plots saved to results_noisy_all/")


if __name__ == "__main__":
    main()
