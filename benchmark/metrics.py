"""
Check by Bank
2025-03-09
"""

"""
Benchmark metrics and ROC/AUC plotting.

Provides:
  - ``compute_all_metrics``  — ML metric dictionary from predictions.
  - ``ResourceTracker``      — context manager for wall-clock timing.
  - ``plot_roc_curve``       — single ROC curve on a matplotlib Axes.
  - ``plot_all_roc_curves``  — overlay all kernels' ROC curves and save.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# -----------------------------------------------------------------------
# ML metrics
# -----------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true  : True binary labels {-1, +1} or {0, 1}.
    y_pred  : Predicted labels.
    y_score : Decision function or probability scores for the positive class.
              Required for AUC computation.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc (if y_score given).
    """
    results: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            results["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            results["roc_auc"] = float("nan")
    return results


def analyze_circuit_resources(qc: QuantumCircuit) -> dict[str, Any]:
    """Transpile a circuit to a common target and extract detailed resource metrics.
    
    Target:
      - basis_gates:   ['cx', 'id', 'rz', 'sx', 'x']
      - opt_level:     1
      - backend/seed:  seed_transpiler=42
    
    Returns a dictionary of:
      - total_depth, two_qubit_depth, total_gates, two_qubit_count, 
        one_qubit_count, gate_breakdown
    """
    # 1. Common transpilation
    basis = ['cx', 'id', 'rz', 'sx', 'x']
    t_qc = transpile(
        qc, 
        basis_gates=basis, 
        optimization_level=1, 
        seed_transpiler=42
    )
    
    # 2. Extract metrics
    ops = dict(t_qc.count_ops())
    
    total_depth = t_qc.depth()
    
    # 2-qubit depth: count only 2Q gates (usually cx/ecr/cz), skip barriers
    two_q_depth = t_qc.depth(
        filter_function=lambda x: x.operation.num_qubits == 2 and x.operation.name not in ['barrier', 'measure']
    )
    
    two_q_count = sum(ops.get(g, 0) for g in ['cx', 'ecr', 'cz'])
    total_gates = sum(ops.values()) - ops.get('barrier', 0) - ops.get('measure', 0)
    one_q_count = total_gates - two_q_count
    
    # Sort and format basis gate breakdown
    breakdown_str = ", ".join(f"{k}:{v}" for k, v in sorted(ops.items()) if k not in ['barrier', 'measure'])
    
    return {
        "total_depth": total_depth,
        "two_qubit_depth": two_q_depth,
        "total_gates": total_gates,
        "two_qubit_count": two_q_count,
        "one_qubit_count": one_q_count,
        "gate_breakdown": breakdown_str,
    }


# -----------------------------------------------------------------------
# Resource tracker
# -----------------------------------------------------------------------

class ResourceTracker:
    """Simple context manager that measures wall-clock execution time."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "ResourceTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start


# -----------------------------------------------------------------------
# ROC curve plotting
# -----------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str,
    ax: plt.Axes,
    color: str | None = None,
) -> None:
    """Plot a single ROC curve onto an existing Axes.

    Parameters
    ----------
    y_true  : True binary labels.
    y_score : Scores for the positive class.
    label   : Legend label (e.g. kernel name).
    ax      : Matplotlib Axes to plot on.
    color   : Line color (optional).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    kwargs = {"label": f"{label} (AUC={auc:.3f})"}
    if color:
        kwargs["color"] = color
    ax.plot(fpr, tpr, lw=2, **kwargs)


def plot_all_roc_curves(
    results: dict[str, dict],
    save_path: str | Path,
    title: str = "ROC Curves — Quantum Kernels",
) -> None:
    """Overlay ROC curves for multiple kernels and save as PNG.

    Parameters
    ----------
    results : dict mapping kernel_name → dict with keys 'y_true' and 'y_score'.
    save_path : File path for saved figure (PNG).
    title : Figure title.
    """
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, (name, data) in enumerate(results.items()):
        try:
            plot_roc_curve(
                data["y_true"],
                data["y_score"],
                label=name,
                ax=ax,
                color=colors[idx % len(colors)],
            )
        except Exception:
            pass  # skip kernels where AUC cannot be computed

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    save_path: str | Path,
) -> None:
    """Save a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {label}", fontsize=12)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
