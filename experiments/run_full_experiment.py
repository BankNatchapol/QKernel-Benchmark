"""
run_full_experiment.py
======================
Master benchmark script for the paper:
  "Quantum Kernel Methods for High Energy Physics"

Implements the FULL experimental design — all configs × all noise regimes:
  - 5 Feature Maps: ZZ, FM1, FM2, FM3, FM4
  - 2 Fixed Kernels: FQK, PQK
  - 1 Trainable Kernel: QKTA
  - 4 Layer settings: L=1,2,3,4
  - 3 Noise Regimes: low (0.5), realistic (1.0), high (2.0)
  - 3 Datasets: higgs, hepmass, energyflow

Total: [(5 FMs × 2 kernels × 4 layers) + (1 QKTA × 4 layers)] × 3 noise regimes
     = 44 configs × 3 regimes = 132 experiment slices per dataset.

All experiments share the same splits, preprocessing, seed, and evaluation
pipeline to ensure fair comparison. Only the noise model changes between regimes.

Usage:
  python experiments/run_full_experiment.py --device torino --samples 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Path Setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Quantum Imports ──────────────────────────────────────────────────────────
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Fake backend providers
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeSherbrooke, FakeKyiv, FakeOsaka
    FAKE_BACKENDS = {
        "torino":    FakeTorino,
        "sherbrooke": FakeSherbrooke,
        "kyiv":      FakeKyiv,
        "osaka":     FakeOsaka,
    }
except ImportError:
    FAKE_BACKENDS = {}

from experiments.run_single_noisy_sim import adjust_noise_model
from feature_maps.zz_map import ZZMap
from feature_maps.custom_maps import CFM1, CFM2, CFM3, CFM4
from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from kernels.trainable_kernel import TrainableKernel
from benchmark.runner import BenchmarkRunner


# ── Noise Regimes ─────────────────────────────────────────────────────────────
NOISE_REGIMES = {
    "low":       0.5,
    "realistic": 1.0,
    "high":      2.0,
}


# ── Kernel Grid Builder ────────────────────────────────────────────────────────

def build_kernel_grid(
    n_qubits: int,
    shots: int,
    layers_list: list[int],
    backend_name: str,
    backend: Any,
    qkta_max_iter: int,
    chunk_size: int,
) -> dict[str, Any]:
    """
    Build the full kernel grid for ONE noise regime.

    Returns a flat dict: {kernel_label → kernel_instance} covering:
      - FQK × {ZZ, FM1-4} × {L=1..4}
      - PQK × {ZZ, FM1-4} × {L=1..4}
      - QKTA × {L=1..4}
    """
    kernels: dict[str, Any] = {}

    feature_map_builders = {
        "ZZ":  ZZMap,
        "FM1": CFM1,
        "FM2": CFM2,
        "FM3": CFM3,
        "FM4": CFM4,
    }

    base_kwargs = dict(
        n_qubits=n_qubits,
        shots=shots,
        backend_name=backend_name,
        backend=backend,
        chunk_size=chunk_size,
    )

    # Fixed kernels: FQK and PQK
    for L in layers_list:
        for fm_name, fm_cls in feature_map_builders.items():
            kernels[f"FQK_{fm_name}_L{L}"] = FidelityKernel(
                **base_kwargs, feature_map=fm_cls(n_qubits=n_qubits, reps=L)
            )
            kernels[f"PQK_{fm_name}_L{L}"] = ProjectedKernel(
                **base_kwargs, feature_map=fm_cls(n_qubits=n_qubits, reps=L)
            )

    # Trainable kernel: QKTA
    for L in layers_list:
        kernels[f"QKTA_L{L}"] = TrainableKernel(
            **base_kwargs, reps=L, max_iter=qkta_max_iter,
        )

    return kernels


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full QKernel experiment suite with 3 noise regimes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--qubits",     type=int,   default=4,    help="Number of qubits / features.")
    parser.add_argument("--samples",    type=int,   default=200,  help="Total samples per dataset.")
    parser.add_argument("--shots",      type=int,   default=1024, help="Shots per circuit.")
    parser.add_argument("--seed",       type=int,   default=42,   help="Global random seed.")
    parser.add_argument("--qkta_iter",  type=int,   default=50,   help="QKTA max optimizer iterations.")
    parser.add_argument("--chunk",      type=int,   default=200,  help="Circuits per submission chunk.")
    parser.add_argument(
        "--device", type=str, default="torino",
        choices=list(FAKE_BACKENDS.keys()),
        help="IBM fake backend to derive noise model from.",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["higgs", "hepmass", "energyflow"],
        help="List of datasets to benchmark.",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+",
        default=[1, 2, 3, 4],
        help="Layer counts to test.",
    )
    parser.add_argument(
        "--regimes", type=str, nargs="+",
        default=["low", "realistic", "high"],
        choices=["low", "realistic", "high"],
        help="Noise regimes to run.",
    )
    parser.add_argument("--out_dir",    type=str,   default="results_full_experiment", help="Output directory.")

    args = parser.parse_args()

    if not FAKE_BACKENDS:
        print("❌ qiskit_ibm_runtime not installed. Cannot load fake backends.")
        return

    backend_class = FAKE_BACKENDS[args.device]

    # ── Summary ────────────────────────────────────────────────────────────────
    n_fm     = 5                                             # ZZ + FM1-4
    n_fixed  = n_fm * 2 * len(args.layers)                  # FMs × (FQK + PQK) × layers
    n_qkta   = len(args.layers)                              # QKTA × layers
    n_configs = n_fixed + n_qkta
    n_regimes = len(args.regimes)

    print(f"\n{'='*70}")
    print(f"  🔬 Quantum Kernel Full Experiment (Noisy AerSim)")
    print(f"{'='*70}")
    print(f"   FakeBackend  : {args.device}")
    print(f"   Noise Regimes: {args.regimes}  (factors: {[NOISE_REGIMES[r] for r in args.regimes]})")
    print(f"   Configs/DS   : {n_fixed} fixed + {n_qkta} QKTA = {n_configs}")
    print(f"   Total slices : {n_configs} × {n_regimes} regimes × {len(args.datasets)} datasets = {n_configs * n_regimes * len(args.datasets)}")
    print(f"   Qubits: {args.qubits}  |  Samples: {args.samples}  |  Shots: {args.shots}  |  Seed: {args.seed}")
    print(f"{'='*70}\n")

    # ── Outer Loop: Noise Regimes ──────────────────────────────────────────────
    all_dfs = []
    t_global = time.perf_counter()

    for regime_name in args.regimes:
        factor = NOISE_REGIMES[regime_name]
        print(f"\n{'─'*70}")
        print(f"  🛰️  REGIME: {regime_name.upper()}  (noise_factor = {factor})")
        print(f"{'─'*70}")

        # Build (fresh) backend for this regime
        real_backend = backend_class()
        base_nm = NoiseModel.from_backend(real_backend)
        scaled_nm = adjust_noise_model(base_nm, factor) if factor != 1.0 else base_nm
        sim_backend = AerSimulator.from_backend(real_backend, noise_model=scaled_nm)

        # Build kernels pointing at this backend
        kernels = build_kernel_grid(
            n_qubits=args.qubits,
            shots=args.shots,
            layers_list=args.layers,
            backend_name="ibm",       # Use ISA-compliant path (SamplerV2 + transpile)
            backend=sim_backend,
            qkta_max_iter=args.qkta_iter,
            chunk_size=args.chunk,
        )

        regime_dir = f"{args.out_dir}/{regime_name}"
        runner = BenchmarkRunner(
            kernels=kernels,
            dataset_names=args.datasets,
            n_qubits=args.qubits,
            shots=args.shots,
            n_samples=args.samples,
            results_dir=regime_dir,
        )

        t0 = time.perf_counter()
        df = runner.run(random_state=args.seed)
        df["regime"] = regime_name
        df["noise_factor"] = factor
        all_dfs.append(df)
        print(f"  ✅ Regime '{regime_name}' done in {(time.perf_counter()-t0)/60:.1f} min")

    # ── Save Consolidated Results ──────────────────────────────────────────────
    all_results = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = Path(args.out_dir) / "full_experiment_results.csv"
    all_results.to_csv(csv_path, index=False)

    elapsed = (time.perf_counter() - t_global) / 60
    display_cols = [c for c in [
        "kernel", "dataset", "regime", "accuracy", "roc_auc", "f1",
        "n_qubits", "total_depth", "2q_count", "wall_clock_s"
    ] if c in all_results.columns]

    print("\n" + "="*70)
    print("  📊 FULL RESULTS SUMMARY")
    print("="*70)
    print(all_results[display_cols].to_string(index=False))
    print(f"\n⏱️  Total time: {elapsed:.1f} min")
    print(f"✅ All results saved to: {csv_path}")


if __name__ == "__main__":
    main()
