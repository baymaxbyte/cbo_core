# CvAdamW: Thermodynamically-Aware Optimization for Grokking

A custom optimizer that monitors the specific heat (Cv) of a Transformer's attention
mechanism in real-time and dynamically injects thermal energy (via weight decay) to
force generalization in models trapped in memorization.

## The Problem

Neural networks exhibit "grokking": they memorize training data instantly, then
waste thousands of epochs before suddenly generalizing. This "critical slowing down"
period is computationally wasteful and unpredictable.

## The Solution

The variance of pre-softmax attention logits (Cv = Var(QK^T/sqrt(d_k))) peaks at
phase transitions. CvAdamW detects this peak and responds with proportional heat
injection, forcing the model across the generalization boundary.

```
lambda_t = base_wd + kappa * max(0, mu_t - tau)
```

Where mu_t is the EMA momentum of Cv velocity.

## Quick Start

```python
from phase4_cv_adamw_exp3 import CvAdamW

optimizer = CvAdamW(
    model.parameters(),
    lr=3e-4,
    base_weight_decay=0.1,
    kappa=5.0,
    tau=0.02,
    alpha=0.9,
)

# Training loop
for epoch in range(epochs):
    logits, cv = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(current_cv=cv, train_accuracy=train_acc)
```

## Results

| Method | Grokking Epoch | Notes |
|--------|---------------|-------|
| Baseline (AdamW, no intervention) | Never (>4000) | Model trapped |
| Step-Function (Phase 4A) | 3333 | Discrete spike at Cv peak |
| **CvAdamW (Phase 4B)** | **2802** | Continuous proportional response |

## Repository Structure

```
experiment_core/
|
|-- phase3_benchmark_exp3.py          # Baseline training + ghost triggers
|-- phase4_training_loop_exp3.py      # Step-function heat injection
|-- phase4_cv_adamw_exp3.py           # CvAdamW optimizer (main contribution)
|-- phase4_cv_adamw_exp3_prefix2.py   # Pre-bugfix version (for comparison)
|
|-- plot_master_figure.py             # 3-panel publication figure
|-- plot_wd_comparison.py             # WD overlay: 4A vs 4B
|-- run_exp3.sh                       # Multi-seed runner (5 seeds, 10k epochs)
|
|-- JOURNEY.md                        # Full development documentation
|-- SUBSTACK_ARTICLE.md               # Narrative article
|
|-- data/                             # Per-seed CSVs and JSONs
|   |-- phase3_exp3_metrics_seed{N}.csv
|   |-- phase3_exp3_benchmark_results_seed{N}.json
|   |-- phase4a_exp3_metrics_seed{N}.csv
|   |-- phase4a_exp3_results_seed{N}.json
|   |-- phase4b_exp3_metrics_seed{N}.csv
|   |-- phase4b_exp3_results_seed{N}.json
|
|-- figures/                          # Per-seed plots
|   |-- phase3_exp3_triggers_seed{N}.png
|   |-- phase3_exp3_kinematics_seed{N}.png
|   |-- phase4a_exp3_intervention_seed{N}.png
|   |-- phase4b_exp3_cv_adamw_seed{N}.png
|   |-- wd_comparison_4a_vs_4b_seed{N}.png
|
|-- scale_invariant/                  # Z-Score reformulation (no kappa/tau)
    |-- phase3_benchmark.py           # Baseline + 5 ghost triggers
    |-- phase4_cv_adamw.py            # Cold-start gate variant
    |-- phase4_cv_adamw_wo_coldstart.py  # Continuous sensor variant
    |-- run_scale_invariant.sh        # 10-seed runner (7000 epochs)
    |-- README.md                     # Detailed design doc
    |-- data/                         # 10 seeds x 3 phases
    |-- figures/                      # All per-seed plots
```

## Architecture

```
Model:       2-layer decoder-only Transformer
d_model:     128
Heads:       4 (d_k = 32)
Position:    RoPE (Rotary Positional Embeddings)
Activation:  GELU
Task:        a + b (mod 97), 50/50 train/val split
```

## Two Formulations

### 1. kappa/tau (Original, Task-Specific)

```
mu_t = alpha * mu_{t-1} + (1-alpha) * (Cv(t) - Cv(t-1))
lambda_t = base_wd + kappa * max(0, mu_t - tau)
```

- kappa=5.0 amplifies momentum into WD (max observed: 2.6)
- tau=0.02 noise floor (ignore small fluctuations)
- Aggressive, effective, but needs re-tuning per task

### 2. Z-Score (Scale-Invariant)

```
Z_t = (v_t - mu_t) / (sigma_t + epsilon)
lambda_t = base_wd + max(0, Z_t - z_thresh)
```

- z_thresh=2.0 (universal statistical constant)
- No kappa, no tau, fully dimensionless
- Gentler response (max WD ~0.9 vs ~2.6)
- Portable across tasks without re-tuning

## Running

```bash
# All phases, 5 seeds, 10k epochs
PYTHONUNBUFFERED=1 nohup bash run_exp3.sh > run_exp3.log 2>&1 &

# Scale-invariant, 10 seeds, 7k epochs
cd scale_invariant
PYTHONUNBUFFERED=1 nohup bash run_scale_invariant.sh > run_scale_invariant.log 2>&1 &

# Single phase, single seed
python phase4_cv_adamw_exp3.py --device cuda --epochs 10000 --seed 42
```

## Key Findings

1. **EMA smoothing (alpha=0.9)** is essential to filter mini-batch noise
2. **Memorization gate (train_acc >= 0.99)** prevents premature triggers
3. **Continuous > discrete**: proportional response beats binary spike
4. **"Slingshot Blinding"**: discrete triggers fail when structural reorganization
   briefly drops train_acc below the gate threshold
5. **Scale-invariance trades strength for universality**: Z-Score version is
   portable but injects less heat than kappa/tau version

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## Citation

Based on the thermodynamic framework from:
> Gunn Kim, "Thermodynamic Isomorphism of Transformers: A Lagrangian Approach
> to Attention Dynamics," arXiv:2602.08216v2 (2026).
