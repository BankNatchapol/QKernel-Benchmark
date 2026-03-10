"""
run_single.py — Run a single quantum kernel experiment from the CLI.

Usage:
    python experiments/run_single.py \
        --kernel fqk \
        --dataset moons \
        --n_qubits 4 \
        --shots 1024 \
        --n_samples 80

Results are printed to stdout and saved as JSON in results/.
"""

from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path

# Allow running from repository root OR experiments/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel
from benchmark.runner import BenchmarkRunner


KERNEL_REGISTRY = {
    "fqk": lambda n_qubits, shots, seed, chunk_size, backend_name: FidelityKernel(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name),
    "pqk": lambda n_qubits, shots, seed, chunk_size, backend_name: ProjectedKernel(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name),
    "qkta": lambda n_qubits, shots, seed, chunk_size, backend_name: TrainableKernel(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name),
    "qflair": lambda n_qubits, shots, seed, chunk_size, backend_name: QFLAIRKernel(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single QKernel-Benchmark experiment."
    )
    parser.add_argument(
        "--kernel",
        choices=list(KERNEL_REGISTRY.keys()),
        default="fqk",
        help="Quantum kernel to use (default: fqk).",
    )
    parser.add_argument(
        "--dataset",
        default="moons",
        help="Dataset name (default: moons).",
    )
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits.")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit.")
    parser.add_argument("--n_samples", type=int, default=80, help="Total dataset size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Circuits per batch for memory-safe simulation.")
    parser.add_argument("--backend_name", choices=["aer", "statevector"], default="aer", help="Backend to use (aer or statevector).")
    parser.add_argument(
        "--results_dir", default="results", help="Directory for output files."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kernel = KERNEL_REGISTRY[args.kernel](args.n_qubits, args.shots, args.seed, args.chunk_size, args.backend_name)

    runner = BenchmarkRunner(
        kernels={args.kernel: kernel},
        dataset_names=[args.dataset],
        n_qubits=args.n_qubits,
        shots=args.shots,
        n_samples=args.n_samples,
        results_dir=args.results_dir,
    )

    df = runner.run(random_state=args.seed)

    # Save JSON alongside CSV
    record = df.iloc[0].to_dict()
    json_path = Path(args.results_dir) / f"{args.kernel}_{args.dataset}.json"
    with open(json_path, "w") as f:
        json.dump(
            {k: v for k, v in record.items() if not isinstance(v, float) or not (v != v)},
            f,
            indent=2,
            default=str,
        )
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
