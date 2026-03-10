from __future__ import annotations

"""
Check by Bank
2025-03-09

BenchmarkRunner: orchestrates the full benchmark grid.

Runs every combination of (kernel × dataset), fits the classifier,
collects ML metrics plus quantum resource stats, and returns a DataFrame.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets.loader import load_dataset
from kernels.base import QuantumKernel
from kernels.trainable_kernel import TrainableKernel
from kernels.qflair_kernel import QFLAIRKernel
from classifiers.qsvm import QSVM
from benchmark.metrics import (
    compute_all_metrics,
    ResourceTracker,
    plot_all_roc_curves,
    plot_confusion_matrix,
)

# ── ANSI helpers (gracefully degrade if terminal doesn't support them) ──
_BOLD  = "\033[1m"
_CYAN  = "\033[36m"
_GREEN = "\033[32m"
_YELLOW= "\033[33m"
_RESET = "\033[0m"


def _header(text: str) -> str:
    return f"{_BOLD}{_CYAN}{text}{_RESET}"


def _ok(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"


def _warn(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"


class BenchmarkRunner:
    """Run a benchmark grid over multiple quantum kernels and datasets.

    Parameters
    ----------
    kernels : dict mapping label (str) → QuantumKernel instance.
    dataset_names : list of dataset name strings.
    n_qubits : number of qubits / features per sample.
    shots : shots per circuit execution.
    n_samples : total samples per dataset.
    C : SVM regularisation parameter.
    results_dir : directory where CSV and plots are saved.
    """

    def __init__(
        self,
        kernels: dict[str, QuantumKernel],
        dataset_names: list[str],
        n_qubits: int = 4,
        shots: int = 1024,
        n_samples: int = 80,
        C: float = 1.0,
        results_dir: str | Path = "results",
    ):
        self.kernels = kernels
        self.dataset_names = dataset_names
        self.n_qubits = n_qubits
        self.shots = shots
        self.n_samples = n_samples
        self.C = C
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------

    def run_one(
        self,
        kernel_name: str,
        kernel: QuantumKernel,
        dataset_name: str,
        random_state: int = 42,
        outer_pbar: tqdm | None = None,
    ) -> dict[str, Any]:
        """Run one kernel × dataset experiment and return a metric dict."""

        def _set_desc(msg: str) -> None:
            if outer_pbar is not None:
                outer_pbar.set_postfix_str(msg, refresh=True)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Load data
        _set_desc("loading data")
        X_train, X_test, y_train, y_test = load_dataset(
            dataset_name,
            n_samples=self.n_samples,
            n_features=self.n_qubits,
            random_state=random_state,
        )

        n_original_features = X_train.shape[1] if len(X_train.shape) > 1 else 1

        # Reshape EnergyFlow (3D array) to 2D
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test  = X_test.reshape(X_test.shape[0], -1)

        # Apply PCA if features > n_qubits
        if X_train.shape[1] > self.n_qubits:
            _set_desc(f"PCA ({X_train.shape[1]} -> {self.n_qubits})")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            pca = PCA(n_components=self.n_qubits, random_state=random_state)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # Final MinMax Scaling to [0, pi] for quantum feature maps
        from sklearn.preprocessing import MinMaxScaler
        scaler_pi = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler_pi.fit_transform(X_train)
        X_test = scaler_pi.transform(X_test)

        row: dict[str, Any] = {
            "kernel": kernel_name,
            "dataset": dataset_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "original_features": n_original_features,
        }

        with ResourceTracker() as tracker:
            # --- Optional training step (QKTA / Q-FLAIR) ---
            if isinstance(kernel, (TrainableKernel, QFLAIRKernel)):
                _set_desc("training kernel params")
                kernel.fit(X_train, y_train)

            # --- Kernel matrices ---
            _set_desc(f"K_train ({len(X_train)}×{len(X_train)})")
            K_train = kernel.build_kernel_matrix(X_train)

            _set_desc(f"K_test  ({len(X_test)}×{len(X_train)})")
            K_test = kernel.build_kernel_matrix(X_test, X_train)

            # --- QSVM ---
            _set_desc("fitting QSVM")
            clf = QSVM(C=self.C)
            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)
            try:
                y_score = clf.predict_proba(K_test)[:, 1]
            except Exception:
                y_score = None

        row["wall_clock_s"] = tracker.elapsed

        # --- ML metrics ---
        ml = compute_all_metrics(y_test, y_pred, y_score)
        row.update(ml)

        # --- Quantum resource metrics ---
        row["n_qubits"]        = kernel.stats.n_qubits
        row["total_depth"]     = kernel.stats.total_depth
        row["2q_depth"]        = kernel.stats.two_qubit_depth
        row["total_gates"]     = kernel.stats.total_gates
        row["2q_count"]        = kernel.stats.two_qubit_count
        row["1q_count"]        = kernel.stats.one_qubit_count
        row["gate_breakdown"]  = kernel.stats.gate_breakdown
        row["total_shots"]     = kernel.stats.total_shots
        row["n_evaluations"]   = kernel.stats.n_evaluations

        # --- Confusion matrix ---
        cm_path = self.results_dir / "plots" / f"cm_{kernel_name}_{dataset_name}.png"
        plot_confusion_matrix(y_test, y_pred, f"{kernel_name} / {dataset_name}", cm_path)

        # Store for ROC overlay
        row["_y_true"]  = y_test
        row["_y_score"] = y_score

        return row

    # ------------------------------------------------------------------
    # Full grid
    # ------------------------------------------------------------------

    def run(self, random_state: int = 42) -> pd.DataFrame:
        """Run the full grid and return a DataFrame of results."""
        records: list[dict] = []
        dataset_roc: dict[str, dict[str, dict]] = {}

        total_jobs = len(self.dataset_names) * len(self.kernels)
        print(
            f"\n{_header('QKernel-Benchmark')}"
            f"  {len(self.kernels)} kernels × {len(self.dataset_names)} datasets"
            f"  ({total_jobs} experiments total)"
            f"  |  qubits={self.n_qubits}  shots={self.shots}  n_samples={self.n_samples}\n"
        )

        with tqdm(
            total=total_jobs,
            desc="Overall",
            unit="exp",
            ncols=88,
            colour="cyan",
            position=0,
            leave=True,
        ) as pbar:
            for dataset_name in self.dataset_names:
                dataset_roc[dataset_name] = {}
                tqdm.write(f"\n{_header('▶ Dataset:')} {dataset_name}")

                for kernel_name, kernel in self.kernels.items():
                    pbar.set_description(f"{kernel_name}/{dataset_name}")

                    t0 = time.perf_counter()
                    row = self.run_one(
                        kernel_name, kernel, dataset_name, random_state, outer_pbar=pbar
                    )
                    elapsed = time.perf_counter() - t0

                    # Pretty one-liner result
                    acc   = row.get("accuracy", float("nan"))
                    f1    = row.get("f1",       float("nan"))
                    auc   = row.get("roc_auc",  float("nan"))
                    depth = row.get("total_depth", "?")
                    q2_c  = row.get("2q_count", "?")
                    q2_d  = row.get("2q_depth", "?")
                    tqdm.write(
                        f"  {_ok('✓')} {kernel_name:<10}"
                        f"  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}"
                        f"  2q_cnt={q2_c}  2q_dep={q2_d}  tot_dep={depth}  {elapsed:.1f}s"
                    )

                    # Save ROC data
                    if row.get("_y_score") is not None:
                        dataset_roc[dataset_name][kernel_name] = {
                            "y_true":  row.pop("_y_true"),
                            "y_score": row.pop("_y_score"),
                        }
                    else:
                        row.pop("_y_true", None)
                        row.pop("_y_score", None)

                    records.append(row)
                    pbar.update(1)

                # ROC overlay per dataset
                if dataset_roc[dataset_name]:
                    roc_path = self.results_dir / "plots" / f"roc_{dataset_name}.png"
                    plot_all_roc_curves(
                        dataset_roc[dataset_name],
                        save_path=roc_path,
                        title=f"ROC Curves — {dataset_name}",
                    )

        df = pd.DataFrame(records)
        csv_path = self.results_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)

        # ── Summary table ──────────────────────────────────────────────
        cols = [
            "kernel", "dataset", "accuracy", "f1", "roc_auc", 
            "n_qubits", "2q_count", "2q_depth", "total_depth", "1q_count", 
            "total_gates", "wall_clock_s"
        ]
        
        # Only select columns that actually exist (in case of partial runs or missing data)
        valid_cols = [c for c in cols if c in df.columns]
        summary = df[valid_cols].copy()
        
        if "accuracy" in summary.columns:
            summary["accuracy"] = summary["accuracy"].map("{:.3f}".format)
        if "f1" in summary.columns:
            summary["f1"] = summary["f1"].map("{:.3f}".format)
        if "roc_auc" in summary.columns:
            summary["roc_auc"] = summary["roc_auc"].map(
                lambda x: f"{x:.3f}" if not (isinstance(x, float) and x != x) else " n/a"
            )
        if "wall_clock_s" in summary.columns:
            summary["wall_clock_s"] = summary["wall_clock_s"].map("{:.1f}s".format)

        # Build supplementary table for gate breakdown if available
        breakdown_table = ""
        if "gate_breakdown" in df.columns:
            bd_summary = df[["kernel", "dataset", "gate_breakdown"]].copy()
            breakdown_table = (
                f"\n{_header('=' * 85)}\n"
                f"{_header('  Gate Breakdown Supplement')}\n"
                f"{_header('=' * 85)}\n"
                f"{bd_summary.to_string(index=False)}\n"
                f"{_header('=' * 85)}"
            )

        print(f"\n{_header('=' * 110)}")
        print(_header("  Results Summary"))
        print(_header("=" * 110))
        print(summary.to_string(index=False))
        print(_header("=" * 110))
        if breakdown_table:
            print(breakdown_table)
        print(f"\n{_ok('✓')} Saved → {csv_path}\n")

        return df
