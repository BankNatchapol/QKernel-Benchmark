# Noise Factor Methodology

This document details the implementation of the `noise_factor` parameter used in the multi-regime noisy simulation suite of the QKernel-Benchmark project.

## Overview

To evaluate quantum kernel robustness, we define three controlled noise regimes by applying global scaling factors to a backend-calibrated noise model. This approach preserves the specific gate- and qubit-dependent asymmetries of real hardware while modulating the overall noise strength.

## Controlled Noise Regimes

We define the following explicit regimes for the study:

| Regime | Factor ($\alpha$) | Description |
| :--- | :--- | :--- |
| **Low** | **0.5** | Scaled backend-calibrated noise (optimistic future hardware). |
| **Realistic** | **1.0** | Unmodified backend-derived baseline (current hardware state). |
| **High** | **2.0** | Scaled backend-calibrated noise (stress-test regime). |

## Implementation Details

The scaling logic treats gate errors and readout errors separately to ensure physical consistency and normalization.

### 1. Scaling Gate Errors
For each gate error entry (e.g., depolarizing noise), the probabilities are stored as a vector $p$ where $p_0$ is the "no error" probability.

1.  **Isolate Error**: Compute total error probability $P_{\mathrm{err}} = \sum_{k>0} p_k$.
2.  **Scale**: $P'_{\mathrm{err}} = \min(\alpha P_{\mathrm{err}}, 0.999)$.
3.  **Renormalize**:
    - $p_k' = p_k \frac{P'_{\mathrm{err}}}{P_{\mathrm{err}}}$ for $k > 0$
    - $p_0' = 1 - P'_{\mathrm{err}}$

### 2. Scaling Readout Errors
Readout errors are defined by a transition matrix $M_{ij} = P(\text{measured } j \mid \text{prepared } i)$.

1.  **Isolate Row Errors**: For each row $i$, the off-diagonal terms ($j \neq i$) are the errors.
2.  **Scale Sum**: Scale the sum of off-diagonal probabilities by $\alpha$.
3.  **Renormalize**:
    - $M_{ij}' = M_{ij} \times \text{scale}$ for $j \neq i$.
    - $M_{ii}' = 1 - \sum_{j \neq i} M_{ij}'$ (ensures the row sum is exactly 1).

## Rebuilding and Simulation
The modified dictionary is reconstructed into a valid Qiskit `NoiseModel` and attached to the `AerSimulator`:

```python
scaled_nm = NoiseModel.from_dict(nm_dict)
backend = AerSimulator.from_backend(real_backend, noise_model=scaled_nm)
```

## Reproducibility
The `run_all_noisy_sim.py` script automatically iterates through all three regimes for all kernels and datasets, saving full metrics and plots into segregated subdirectories:
- `results_noisy_all/low/`
- `results_noisy_all/realistic/`
- `results_noisy_all/high/`
- `results_noisy_all/full_regime_summary.csv`

## Sanity Checks
The implementation includes automated checks to verify:
- Every probability vector and readout row sums exactly to 1.0.
- No negative probabilities appear.
- The simulator accepts the final model.

---
*Note: These regimes are intended as controlled robustness settings for comparative analysis, as backend-derived models are approximations of device behavior.*
