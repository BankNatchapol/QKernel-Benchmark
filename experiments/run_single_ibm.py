import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel
from benchmark.runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark on IBM Quantum hardware.")
    parser.add_argument("--kernel", type=str, default="fidelity", choices=["fidelity", "projected", "trainable", "qflair"], help="Kernel type.")
    parser.add_argument("--dataset", type=str, default="circles", help="Dataset name.")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits/features.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples (keep small for real hardware!).")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit.")
    parser.add_argument("--device", type=str, default="ibm_torino", help="IBM Quantum device name.")
    parser.add_argument("--chunk", type=int, default=100, help="Circuits per job (chunk size).")
    
    args = parser.parse_args()

    # 1. Authenticate
    load_dotenv()
    api_key = os.getenv("IBM_API_KEY")
    if not api_key:
        print("❌ Error: IBM_API_KEY not found in .env file.")
        return

    print(f"✅ Authenticating with IBM Quantum for device: {args.device}...")
    service = QiskitRuntimeService(token=api_key)
    # Optional: Trigger discovery to avoid lazy-loading issues
    print("Discovering backends...")
    _ = service.backends()
    print(f"Connecting to {args.device}...")
    backend = service.backend(args.device)

    # 2. Initialize Kernel
    print(f"Initializing {args.kernel} kernel for {args.qubits} qubits...")
    if args.kernel == "fidelity":
        kernel = FidelityKernel(n_qubits=args.qubits, shots=args.shots, backend_name="ibm", backend=backend, chunk_size=args.chunk)
    elif args.kernel == "projected":
        kernel = ProjectedKernel(n_qubits=args.qubits, shots=args.shots, backend_name="ibm", backend=backend, chunk_size=args.chunk)
    elif args.kernel == "trainable":
        # Note: Trainable on real hardware will be VERY slow due to iterations
        kernel = TrainableKernel(n_qubits=args.qubits, shots=args.shots, backend_name="ibm", backend=backend, chunk_size=args.chunk, max_iter=5)
    elif args.kernel == "qflair":
        # Note: QFLAIR on real hardware will be VERY slow due to greedy growth
        kernel = QFLAIRKernel(n_qubits=args.qubits, shots=args.shots, backend_name="ibm", backend=backend, chunk_size=args.chunk, n_layers=2)

    # 3. Run Experiment
    runner = BenchmarkRunner(
        kernels={args.kernel: kernel},
        dataset_names=[args.dataset],
        n_qubits=args.qubits,
        shots=args.shots,
        n_samples=args.samples,
        results_dir="results_ibm"
    )

    print(f"\n🚀 Starting benchmark: {args.kernel} on {args.dataset} ({args.samples} samples) using {args.device}")
    results = runner.run_one(args.kernel, kernel, args.dataset)
    
    print("\n=== Experiment Results ===")
    for k, v in results.items():
        if k not in ["confusion_matrix", "roc_curve"]:
            print(f"{k}: {v}")

    print(f"\n✅ Results saved to results_ibm/")

if __name__ == "__main__":
    main()
