# QKernel-Benchmark

A clean, modular benchmarking suite for **4 quantum kernel methods** applied to binary classification. Built on [Qiskit](https://qiskit.org/) and [scikit-learn](https://scikit-learn.org/), runs entirely on the Aer simulator.

---

## Kernels

| Kernel | Description | Reference |
|--------|-------------|-----------|
| **FQK** | Fidelity Quantum Kernel — `k(x,x') = \|⟨ϕ(x)\|ϕ(x')⟩\|²` via adjoint circuit | [Havlíček et al., Nature 567, 2019](https://www.nature.com/articles/s41586-019-0980-2) |
| **PQK** | Projected Quantum Kernel — RBF over local 1-qubit reduced density matrices | [Huang et al., Nature Comms 12, 2021](https://www.nature.com/articles/s41467-021-22539-9) |
| **QKTA** | Trainable Kernel via Kernel-Target Alignment — parameterized feature map optimized to align kernel with labels | [Hubregtsen et al., arXiv:2105.02276](https://arxiv.org/abs/2105.02276) |
| **Q-FLAIR** | Feature-Map Architecture Learning — greedy gate selection by KTA, evaluated as FQK | [Barbosa et al., arXiv:2311.04965](https://arxiv.org/abs/2311.04965) |

---

## Datasets

| Name | Type | Notes |
|------|------|-------|
| `ad_hoc` | Synthetic | Quantum-hard random Pauli parity labels |
| `moons` | Synthetic | sklearn `make_moons` |
| `circles` | Synthetic | sklearn `make_circles` |
| `blobs` | Synthetic | sklearn `make_blobs` (2-class) |
| `iris` | Real | Classes 0 vs 1 |
| `wine` | Real | Classes 0 vs 1 |
| `breast_cancer` | Real | Binary classification |

All datasets: PCA → `n_qubits` features, MinMaxScaler → `[0, π]`.

---

## Metrics

**ML:** Accuracy, Precision, Recall, F1-score, ROC-AUC  
**Quantum Resources:** Qubit count, circuit depth, total shots, kernel evaluations, wall-clock time  
**Plots:** ROC curve per experiment + overlay of all kernels

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Single experiment
```bash
python experiments/run_single.py \
    --kernel fqk \
    --dataset moons \
    --n_qubits 4 \
    --shots 1024 \
    --n_samples 80
```

### Full benchmark grid
```bash
python experiments/run_all.py
```

Results saved to:
- `results/benchmark_results.csv` — all metrics in tabular form
- `results/plots/roc_*.png` — ROC curve figures

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
QKernel-Benchmark/
├── requirements.txt
├── README.md
├── Plan.md
├── feature_maps/
│   ├── base.py [✓]               ← Abstract FeatureMap
│   ├── iqp_map.py [✓]            ← IQP encoding
│   ├── zz_map.py [✓]             ← ZZ-FeatureMap
│   └── rx_map.py [✓]             ← Rx angle encoding
├── kernels/
│   ├── base.py [✓]               ← Abstract QuantumKernel
│   ├── fidelity_kernel.py [✓]    ← FQK
│   ├── projected_kernel.py [✓]   ← PQK
│   ├── trainable_kernel.py [✓]   ← QKTA
│   └── qflair_kernel.py [✓]      ← Q-FLAIR
├── classifiers/
│   ├── qsvm.py [✓]               ← Quantum SVM
│   └── qkrr.py [ ]               ← Quantum KRR
├── datasets/
│   └── loader.py [✓]             ← Unified dataset loader
├── benchmark/
│   ├── metrics.py [✓]            ← ML + resource metrics + ROC plots
│   └── runner.py [✓]             ← BenchmarkRunner
├── experiments/
│   ├── run_single.py [✓]
│   └── run_all.py [✓]
└── tests/
    ├── test_kernels.py [ ]
    ├── test_datasets.py [ ]
    └── test_classifiers.py [ ]
```
