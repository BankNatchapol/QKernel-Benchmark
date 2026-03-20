# Methods and Experimental Setup — Paper Notes
> Generated from codebase analysis of QKernel-Benchmark.
> Use these details to fill in the LaTeX sections.

---

## Section: Methods

### Quantum Kernel Evaluation

#### Overview

All four quantum kernel methods share a common structure: classical input data $x \in \mathbb{R}^n$ is encoded into an $n$-qubit quantum state $|\phi(x)\rangle = U(x)|0\rangle^{\otimes n}$ via a parameterized unitary $U(x)$ (the *feature map*). The kernel value between two samples is then derived from the overlap between their encoded quantum states.

The number of qubits $n$ is set equal to the number of input features after preprocessing. All circuits are constructed and simulated using Qiskit 2.3.0.

---

#### 1. Fidelity Quantum Kernel (FQK)

**Definition.** The FQK computes the squared inner product between two encoded quantum states:

$$k_\text{FQK}(x, x') = |\langle\phi(x') | \phi(x)\rangle|^2$$

**Feature map.** Both FQK and PQK use the **ZZFeatureMap** from Qiskit Machine Learning (Havlíček et al., 2019, *Nature* 567, 209–212), with 2 repetitions (`reps=2`). This circuit encodes each feature $x_i$ into rotation angles and applies data-dependent ZZ-interaction entangling gates. The resulting states lie in an exponentially large Hilbert space that is inaccessible to classical computation at large $n$.

**Kernel estimation.** The kernel value is estimated via the *adjoint overlap circuit*:
$$U^\dagger(x') \cdot U(x) |0\rangle^{\otimes n}$$
and measuring the probability of the all-zero bitstring $|0\rangle^{\otimes n}$. This probability equals the squared fidelity by construction. Each kernel entry is estimated from **1024 shots** using `qiskit_aer.AerSimulator`.

**Kernel matrix.** The full $N \times N$ symmetric training kernel matrix is computed by evaluating all $\binom{N}{2}$ unique pairs. The diagonal is set to 1. Circuits are submitted in chunks of up to 4096 to the Aer backend for efficient batching.

**PSD enforcement.** After assembly, a projection to the nearest positive semi-definite (PSD) matrix is applied via eigenvalue clipping: any negative eigenvalue is set to zero and the matrix is reconstructed. Diagonal entries are then forced to exactly 1.0. This ensures the validity of the kernel for SVM training.

**Alternative statevector path.** When the `statevector` backend is selected, the statevector $|\phi(x)\rangle$ is computed exactly for all training points. The kernel matrix is then computed as:
$$K = |V_X \cdot V_X^\dagger|^2, \quad V_X \in \mathbb{C}^{N \times 2^n}$$
using a single dense matrix multiplication, making it exact (no shot noise) and substantially faster for large $N$.

---

#### 2. Projected Quantum Kernel (PQK)

**Definition.** Instead of measuring the full state overlap, PQK projects the quantum state onto a set of local single-qubit observables and applies a classical RBF kernel to the resulting vectors.

**Feature map.** The same ZZFeatureMap with `reps=2` is used as for FQK. The encoded statevector $|\phi(x)\rangle$ is computed exactly using `qiskit.quantum_info.Statevector`.

**Projection.** For each qubit $q = 0, \ldots, n-1$, the single-qubit reduced density matrix $\rho_q(x) = \text{Tr}_{-q}[|\phi(x)\rangle\langle\phi(x)|]$ is obtained by partial tracing. The Bloch vector coordinates are extracted:
$$b_q(x) = \big(\langle X_q \rangle, \langle Y_q \rangle, \langle Z_q \rangle\big) = \big(\text{Tr}[\rho_q X], \text{Tr}[\rho_q Y], \text{Tr}[\rho_q Z]\big)$$

All $n$ qubit Bloch vectors are concatenated into a classical feature vector $b(x) \in \mathbb{R}^{3n}$.

**Kernel.** A classical Gaussian (RBF) kernel is applied to the projected features:
$$k^\text{PQ}(x, x') = \exp\!\Big(-\gamma \, \|b(x) - b(x')\|^2\Big), \quad \gamma = 1.0$$

This approach avoids shot noise entirely since the statevector is computed exactly, and is inherently valid as a positive semi-definite kernel (being a composition of a positive-definite map and an RBF kernel).

---

#### 3. Trainable Quantum Kernel via Kernel-Target Alignment (QKTA)

**Motivation.** Fixed feature maps are not necessarily well-suited to a given dataset. QKTA learns the parameters of the feature map to maximize alignment between the kernel matrix and the target similarity structure of the labels (Hubregtsen et al., 2022, *Phys. Rev. A* 106, 042431).

**Feature map ansatz.** A parameterized hardware-efficient ansatz $U(x; \theta)$ is constructed with `reps=1` layers:
- For each qubit $i = 0, \ldots, n-1$: rotate by $R_Y(x_i + \theta_i)$
- Apply a linear chain of CNOT gates: $CX_{0,1}, CX_{1,2}, \ldots, CX_{n-2,n-1}$

The total number of trainable parameters is $n \times \texttt{reps} = n$.

**Training objective.** Parameters are optimized to maximize Kernel-Target Alignment (KTA):
$$A(K_\theta, yy^\top) = \frac{\langle K_\theta,\, yy^\top \rangle_F}{\|K_\theta\|_F \cdot \|yy^\top\|_F}$$
where $y \in \{-1,+1\}^N$ is the label vector, $yy^\top$ is the ideal kernel induced by labels, and $\langle \cdot, \cdot \rangle_F$ denotes the Frobenius inner product.

**Optimization.** The L-BFGS-B gradient-based optimizer (`scipy.optimize.minimize`) is used with a maximum of 30 iterations (`max_iter=30`). Parameters are initialized randomly from $\mathcal{U}(0, 2\pi)$ using a fixed seed. The full training kernel matrix $K_\theta$ is recomputed at each optimization step.

**Kernel evaluation.** After training, a standard fidelity kernel is computed using the optimized $\theta^*$ via the same adjoint overlap circuit procedure as FQK, with 1024 shots.

---

#### 4. Q-FLAIR

**Motivation.** Q-FLAIR (Haas et al., 2024, arXiv:2510.03389) addresses the high cost of trainable kernels by replacing gradient-based full-circuit optimization with a resource-efficient greedy gate selection procedure. Rather than training a pre-defined ansatz, Q-FLAIR builds the feature map from scratch, adding one gate at a time.

**Greedy algorithm.** Given training data $X \in \mathbb{R}^{N \times d}$ and labels $y$, the algorithm proceeds as follows for $L = 6$ layers (maximum):

1. **Initialize.** Start with an empty learned circuit $U_0(x) = \mathbb{I}$.
2. **For each greedy layer:**
   - **Compute base states.** Evaluate the statevectors $\{|\phi(x_i)\rangle\}_{i=1}^N$ for the current learned circuit.
   - **For each gate candidate** from the pool $\{R_z, R_{xx}, R_{yy}, R_{zz}\}$ over all single/two-qubit wire pairs:
     - **Three-point reconstruction.** Probe the candidate gate appended at three angles $\{\alpha_0, \alpha_0 + \pi/2, \alpha_0 - \pi/2\}$ and reconstruct the pairwise surrogate kernel as a cosine function:
       $$k_{ij}(\alpha) = a_{ij}\cos(\alpha - b_{ij}) + c_{ij}$$
       where the coefficients $a_{ij}, b_{ij}, c_{ij}$ are recovered analytically from the three samples.
     - **Classical weight search.** For each feature index $k = 1,\ldots,d$, substitute $\alpha_{ij} = w \cdot (x_{i,k} - x_{j,k})$ and search for the scalar weight $w^*$ that maximizes KTA over the surrogate kernel, using an 8-point coarse grid scan followed by bounded scalar minimization.
   - **Accept** the best (gate, feature index, weight) triple.
   - **Exact refresh.** Recompute the exact sampled kernel matrix of the updated learned circuit and update the stored KTA baseline.
   - **Early stopping.** If the KTA gain is below `min_gain`, stop early.

**Gate candidate pool.** The implementation uses the gate set $\{R_z, R_{xx}, R_{yy}, R_{zz}\}$ applied to single-qubit or two-qubit wire combinations, giving 87 total candidates for $n = 10$ qubits.

**Statevector acceleration.** In the statevector backend, base states are precomputed once per layer using NumPy's `einsum` to apply the candidate gate tensor to all encoded states simultaneously, avoiding repeated Qiskit circuit construction.

**Final kernel.** The fidelity kernel built from the learned feature map $U^*(x)$ is used for classification, evaluated by the same adjoint overlap + all-zero measurement procedure as FQK (Aer backend) or by direct statevector matrix multiplication (statevector backend).

---

### Training Procedure

#### Data Splitting

For **HEPMASS**, the official pre-split train and test CSV files are used directly. For all other datasets, a stratified random split with a **test fraction of 20%** (`test_size=0.2`) is applied, using **random seed 42** for reproducibility.

Sample sizes used in this benchmark: the number of training samples is $0.8 \times N_\text{total}$ and the test samples constitute the remaining $0.2 \times N_\text{total}$.

#### Preprocessing Pipeline

The following identical pipeline is applied to **all kernels and all datasets** before any quantum circuit evaluation:

| Step | Operation | Details |
|------|-----------|---------|
| 1 | **Flatten** (EnergyFlow only) | 3D jet arrays $(N, P, 4)$ are flattened to 2D as $(N, P \times 4)$ |
| 2 | **Standardization** (if PCA needed) | `StandardScaler`: zero mean, unit variance per feature |
| 3 | **PCA** (if features > n_qubits) | `sklearn.decomposition.PCA(n_components=n_qubits)` retaining top principal components |
| 4 | **Qubit clamping** (if features < n_qubits) | Effective `n_qubits` set to actual feature count; user is warned |
| 5 | **MinMax scaling** | `MinMaxScaler(feature_range=(0, π))` scales all features to $[0, \pi]$ |

The PCA transform is fitted on training data only and applied to test data via the same transformation, preventing data leakage.

#### Downstream Classifier

After quantum kernel matrix computation, a **Support Vector Machine** (`sklearn.svm.SVC`) with a precomputed kernel is trained. Probability estimates are enabled (`probability=True`), allowing ROC-AUC computation. The regularization parameter is fixed at **$C = 1.0$** across all experiments. No inner cross-validation or hyperparameter search is performed.

The training flow per experiment is:
1. *(QKTA / Q-FLAIR only)* Fit the kernel parameters on `(X_train, y_train)`
2. Compute `K_train` — the $N_\text{train} \times N_\text{train}$ symmetric kernel matrix
3. Compute `K_test` — the $N_\text{test} \times N_\text{train}$ rectangular kernel matrix
4. Fit `QSVM(C=1.0)` on `K_train, y_train`
5. Predict labels and probability scores on `K_test`
6. Compute all metrics

All random seeds (data split, PCA, kernel initialization, shot-based simulation) are set to **42**.

---

## Section: Experimental Setup

### Software Environment

All experiments are simulation-based. No physical quantum hardware is used. The complete software stack is:

| Component | Library | Version |
|-----------|---------|---------|
| Language | Python | 3.14 |
| Quantum simulation | Qiskit | 2.3.0 |
| Noisy/shot simulation | qiskit-aer | 0.17.2 |
| Quantum ML utilities | qiskit-machine-learning | 0.9.0 |
| Classical ML | scikit-learn | 1.8.0 |
| Numerical computation | numpy | 2.4.2 |
| Scientific optimization | scipy | 1.17.1 |
| HEP jet dataset | EnergyFlow | 1.4.0 |
| HEP dataset access | ucimlrepo | 0.0.7 |
| HDF5 I/O | h5py | 3.16.0 |

**Simulation backend.** Two backends are available:
- **Aer backend** (`qiskit_aer.AerSimulator`): Shot-based simulation with 1024 shots per kernel entry. Includes statistical (shot) noise. Circuits are batched via `transpile()` with `optimization_level=0` for faithful circuit profiling.
- **Statevector backend** (`qiskit.quantum_info.Statevector`): Exact wavefunction simulation. No shot noise; kernel values are exact inner products. Substantially faster for $N \gg 1$ due to matrix operations.

All experiments reported in this paper use the statevector backend unless otherwise specified, to avoid conflating shot noise with algorithmic performance.

**Hardware.** Experiments are run on a MacBook with Apple Silicon (M-series) CPU. No GPU or HPC cluster is used.

---

### Evaluation Metrics

#### Classification Performance

The following metrics are computed on the test split using `sklearn.metrics`:

| Metric | Definition | Notes |
|--------|-----------|-------|
| **Accuracy** | $\frac{\text{correct predictions}}{N_\text{test}}$ | Primary comparison metric |
| **F1-score** | $\frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$ | Macro-averaged; robust to label imbalance |
| **Precision** | $\frac{TP}{TP + FP}$ | Positive-class precision |
| **Recall** | $\frac{TP}{TP + FN}$ | Positive-class recall |
| **ROC-AUC** | Area under receiver operating characteristic curve | Computed from SVM probability estimates via Platt scaling |

ROC-AUC is the **primary comparison criterion** as it evaluates ranking quality independently of classification threshold and is less sensitive to class imbalance.

#### Quantum Circuit Resource Metrics

Resource metrics reflect the hardware cost of executing the feature map circuit (not the overlap circuit), measured after transpilation to the standard basis $\{CX, R_Z, SX, X, \text{id}\}$ using Qiskit's transpiler at `optimization_level=1` with `seed_transpiler=42`:

| Metric | Description |
|--------|-------------|
| **Total depth** | Longest dependency chain in the transpiled circuit |
| **Two-qubit depth** | Circuit depth counting only 2-qubit gates ($CX$) |
| **Total gate count** | Sum of all gates (excluding barriers and measurements) |
| **Two-qubit gate count** | Number of $CX$ (or equivalent) gates |
| **One-qubit gate count** | Total gates minus two-qubit gates |
| **Gate breakdown** | Per-gate-type counts (e.g., `cx:9, rz:20, sx:20`) |

Resource metrics are computed once per kernel and reported as representative values for the feature map circuit evaluated at the first training sample.

#### Wall-Clock Time

Wall-clock time (in seconds) is measured for the complete inference pipeline per experiment: kernel parameter training (QKTA/Q-FLAIR), kernel matrix construction, SVM fitting, and prediction. This is tracked with `time.perf_counter()` for high resolution.
