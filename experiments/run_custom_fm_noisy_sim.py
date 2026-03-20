import os
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

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
from feature_maps.custom_maps import CFM1, CFM2, CFM3, CFM4
from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from benchmark.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run benchmarking for 4 custom feature maps across regimes and layers.")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits/features.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples per experiment.")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit.")
    parser.add_argument("--device", type=str, default="torino", choices=list(FAKE_BACKENDS.keys()), help="IBM device to mimic.")
    parser.add_argument("--chunk", type=int, default=100, help="Circuits per job.")
    
    args = parser.parse_args()

    # 1. Define configurations
    noise_regimes = {
        "low": 0.5,
        "realistic": 1.0,
        "high": 2.0
    }

    datasets = ["higgs", "hepmass", "energyflow"]
    
    custom_fms = {
        "CFM1": CFM1,
        "CFM2": CFM2,
        "CFM3": CFM3,
        "CFM4": CFM4
    }

    reps_list = [1, 2, 3, 4]
    
    final_dfs = []

    print(f"🚀 Starting Custom Feature Map Noisy Benchmark mimicking: {args.device}")
    
    # 2. Outer Loop: Noise Regimes
    for regime_name, factor in noise_regimes.items():
        print(f"\n{'='*60}")
        print(f"🛰️ REGIME: {regime_name.upper()} (factor={factor})")
        print(f"{'='*60}")
        
        # Setup Backend for this regime
        backend_class = FAKE_BACKENDS.get(args.device)
        real_backend = backend_class()
        noise_model = NoiseModel.from_backend(real_backend)
        if factor != 1.0:
            noise_model = adjust_noise_model(noise_model, factor)
        sim_backend = AerSimulator.from_backend(real_backend, noise_model=noise_model)

        # 3. Middle Loop: Repetitions (Layers)
        for reps in reps_list:
            print(f"\n--- 🧱 Layers/Reps: {reps} ---")
            
            # 4. Inner Loop: Kernels and Custom FMs
            kernels = {}
            for fm_name, fm_class in custom_fms.items():
                # Note: reps parameter varies by FM requested logic
                # For FM2, it's internal layers. For others, it's block repetitions.
                # Our CustomFeatureMap class handles reps uniformly by repeating the block.
                fm_instance = fm_class(n_qubits=args.qubits, reps=reps)
                
                kernel_kwargs = {
                    "n_qubits": args.qubits,
                    "shots": args.shots,
                    "backend_name": "ibm",
                    "backend": sim_backend,
                    "chunk_size": args.chunk,
                    "feature_map": fm_instance
                }
                
                kernels[f"FQK_{fm_name}_L{reps}"] = FidelityKernel(**kernel_kwargs)
                kernels[f"PQK_{fm_name}_L{reps}"] = ProjectedKernel(**kernel_kwargs)

            # 5. Run Suite for this (regime, reps) slice
            slice_results_dir = f"results_custom_fm/{regime_name}/L{reps}"
            runner = BenchmarkRunner(
                kernels=kernels,
                dataset_names=datasets,
                n_qubits=args.qubits,
                shots=args.shots,
                n_samples=args.samples,
                results_dir=slice_results_dir
            )

            results_df = runner.run()
            results_df["regime"] = regime_name
            results_df["layers"] = reps
            final_dfs.append(results_df)

    # 6. Final Data Consolidation
    if final_dfs:
        all_results_df = pd.concat(final_dfs, ignore_index=True)
        os.makedirs("results_custom_fm", exist_ok=True)
        all_results_df.to_csv("results_custom_fm/custom_fm_full_benchmark.csv", index=False)
        
        print("\n" + "="*40)
        print("🏆 CUSTOM FEATURE MAP BENCHMARK COMPLETE")
        print("="*40)
        print(f"\n✅ All summaries saved to results_custom_fm/")
    else:
        print("❌ No experiments were run.")

if __name__ == "__main__":
    main()
