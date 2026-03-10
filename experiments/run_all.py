"""
run_all.py — Run the full benchmark grid (all kernels × all datasets).

Results are saved to results/benchmark_results.csv and
per-dataset ROC curves to results/plots/.

Usage:
    python experiments/run_all.py [--n_qubits 4] [--shots 512] [--n_samples 60]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel
from datasets.loader import DATASET_NAMES
from benchmark.runner import BenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full QKernel-Benchmark grid."
    )
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--backend_name", choices=["aer", "statevector"], default="aer")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_NAMES,
        help="Subset of datasets to run (default: all).",
    )
    return parser.parse_args()


def build_kernels(n_qubits: int, shots: int, seed: int, chunk_size: int, backend_name: str) -> dict:
    return {
        "FQK":    FidelityKernel(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name),
        "PQK":    ProjectedKernel(n_qubits=n_qubits, shots=shots, seed=seed, chunk_size=chunk_size, backend_name=backend_name),
        "QKTA":   TrainableKernel(n_qubits=n_qubits, shots=shots, seed=seed, max_iter=30, chunk_size=chunk_size, backend_name=backend_name),
        "Q-FLAIR": QFLAIRKernel(n_qubits=n_qubits, shots=shots, seed=seed, n_layers=3, chunk_size=chunk_size, backend_name=backend_name),
    }


def main() -> None:
    args = parse_args()
    kernels = build_kernels(args.n_qubits, args.shots, args.seed, args.chunk_size, args.backend_name)

    runner = BenchmarkRunner(
        kernels=kernels,
        dataset_names=args.datasets,
        n_qubits=args.n_qubits,
        shots=args.shots,
        n_samples=args.n_samples,
        results_dir=args.results_dir,
    )

    df = runner.run(random_state=args.seed)
    print("\n=== Benchmark Summary ===")
    print(df[["kernel", "dataset", "accuracy", "f1", "roc_auc",
              "2q_count", "2q_depth", "total_depth", "1q_count", "wall_clock_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
