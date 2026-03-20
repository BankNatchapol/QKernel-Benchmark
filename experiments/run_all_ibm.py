import os
import argparse
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel
from benchmark.runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite on IBM Quantum hardware.")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits/features.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples (keep small!).")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit.")
    parser.add_argument("--device", type=str, default="ibm_torino", help="IBM Quantum device name.")
    parser.add_argument("--chunk", type=int, default=100, help="Circuits per job.")
    
    args = parser.parse_args()

    # 1. Authenticate
    load_dotenv()
    api_key = os.getenv("IBM_API_KEY")
    if not api_key:
        print("❌ Error: IBM_API_KEY not found in .env file.")
        return

    print(f"✅ Authenticating with IBM Quantum for device: {args.device}...")
    try:
        service = QiskitRuntimeService(token=api_key)
        _ = service.backends()
        backend = service.backend(args.device)
    except Exception as e:
        print(f"❌ Failed to connect to IBM Quantum: {e}")
        return

    # 2. Define Kernels
    # We exclude Q-FLAIR by default as it's too heavy for hardware benchmarks unless requested
    kernels = {
        "FQK": FidelityKernel(n_qubits=args.qubits, shots=args.shots, backend_name="ibm", backend=backend, chunk_size=args.chunk),
        "PQK": ProjectedKernel(n_qubits=args.qubits, shots=args.shots, backend_name="ibm", backend=backend, chunk_size=args.chunk),
        # QKTA and Q-FLAIR are commented out for real hardware to save time/credits, 
        # but can be re-enabled if needed.
        # "QKTA": TrainableKernel(n_qubits=args.qubits, shots=args.shots, backend_name="ibm", backend=backend, chunk_size=args.chunk, max_iter=5),
    }

    datasets = ["circles", "moons", "breast_cancer"]

    # 3. Run Benchmark
    runner = BenchmarkRunner(
        kernels=kernels,
        dataset_names=datasets,
        n_qubits=args.qubits,
        shots=args.shots,
        n_samples=args.samples,
        results_dir="results_ibm"
    )

    print(f"\n🚀 Starting full IBM benchmark suite on {args.device}")
    print(f"Running on datasets: {datasets}")
    
    results_df = runner.run()
    
    print("\n=== Benchmark Summary ===")
    print(results_df.to_string())

    print(f"\n✅ All results and plots saved to results_ibm/")

if __name__ == "__main__":
    main()
