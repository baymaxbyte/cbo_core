# Scale-Invariant CBO Experiment

## Overview

This experiment replaces the arbitrary hyperparameters (κ, τ) from the original CvAdamW optimizer with a **scale-invariant Z-Score formulation**. Instead of tuning absolute magnitude thresholds, we treat the thermodynamic velocity as a random variable and use its statistical properties to detect phase transitions.

## Core Idea

The thermodynamic velocity $v_t = C_v(t) - C_v(t-1)$ is tracked via Exponential Moving Average (EMA) statistics:

$$\mu_t = \beta_z \cdot \mu_{t-1} + (1 - \beta_z) \cdot v_t$$

$$\sigma_t^2 = \beta_z \cdot \sigma_{t-1}^2 + (1 - \beta_z) \cdot (v_t - \mu_{t-1}) \cdot (v_t - \mu_t)$$

$$Z_t = \frac{v_t - \mu_t}{\sqrt{\sigma_t^2} + \epsilon}$$

The dynamic weight decay becomes:

$$\lambda_t = \lambda_{\text{base}} + \max(0,\ Z_t - z_{\text{thresh}})$$

This is **dimensionless** — no kappa, no tau. Only a universal statistical constant ($z_{\text{thresh}} = 2.0$) identifies when a true phase transition is occurring.

## Experiment Design

- **Task**: Modular addition $(a + b) \mod 97$
- **Model**: 2-layer Transformer with RoPE (same as Phase 2 baseline)
- **Epochs**: 7000
- **Seeds**: 42, 123, 256, 512, 1024, 2048, 3141, 4096, 7777, 9999
- **Z-Score Threshold**: 2.0
- **EMA Smoothing** ($\beta_z$): 0.9
- **Base Weight Decay**: 0.1

## Two Variants of CvAdamW

### Variant A: Cold-Start Gate (`phase4_cv_adamw.py`)

- EMA statistics (μ, σ²) only start tracking **after** `train_accuracy >= 0.99`
- Before that: `dynamic_wd = base_wd`, `z_score = 0`
- **Risk**: Optimizer has zero statistical memory when the gate opens — it starts computing variance cold at the exact moment it needs to detect anomalies

### Variant B: Continuous Sensor / No Cold-Start (`phase4_cv_adamw_wo_coldstart.py`)

- EMA statistics track continuously **from epoch 1**
- The actuator (weight decay injection) is still gated: only applies `max(0, Z - z_thresh)` after `train_accuracy >= 0.99`
- **Advantage**: By the time the gate opens, the optimizer has a warm baseline of what "normal" velocity looks like, so a genuine phase transition spike produces a reliable z-score

### Key Difference

| Component | Cold-Start | Continuous Sensor |
|-----------|-----------|-------------------|
| Sensor (μ, σ² tracking) | Starts at 99% train acc | Starts at epoch 1 |
| Actuator (WD injection) | Starts at 99% train acc | Starts at 99% train acc |
| Statistical memory at gate-open | Zero (cold) | Warm (hundreds of epochs) |

## Phase 3: Trigger Benchmarking

Phase 3 trains the model with **no intervention** (standard AdamW, fixed wd=0.1) and simultaneously runs 5 ghost trigger mechanisms:

1. **Profiler** — Static threshold: fires when velocity > τ
2. **Z-Score (Batch)** — Population z-score over full velocity history
3. **Kinematic** — Second derivative: v > 0 and a < 0 (crest detection)
4. **EMA Z-Score (Continuous)** — Simulates the `_wo_coldstart` variant's trigger point
5. **EMA Z-Score (Gated)** — Simulates the cold-start variant's trigger point

This tells you *when* each optimizer variant would have started injecting heat, measured on the undisturbed baseline trajectory.

## Files

| File | Description |
|------|-------------|
| `phase3_benchmark.py` | Baseline training + 5 ghost triggers |
| `phase4_cv_adamw.py` | Scale-invariant CvAdamW (cold-start gate) |
| `phase4_cv_adamw_wo_coldstart.py` | Scale-invariant CvAdamW (continuous sensor) |
| `phase4_training_loop.py` | Legacy Phase 4A (binary spike/cooldown, not used) |
| `run_scale_invariant.sh` | Runs all 3 scripts across 10 seeds |

## Running

```bash
cd scale_invariant
chmod +x run_scale_invariant.sh
./run_scale_invariant.sh
```

## Output Structure

```
data/
  phase3_seed{N}_metrics.csv              # Baseline metrics per epoch
  phase3_seed{N}_benchmark_results.json   # All 5 trigger epochs + physics logs
  phase4b_seed{N}_metrics.csv             # Cold-start variant metrics
  phase4b_seed{N}_results.json            # Cold-start grokking results
  phase4b_wocs_seed{N}_metrics.csv        # Continuous sensor variant metrics
  phase4b_wocs_seed{N}_results.json       # Continuous sensor grokking results

figures/
  phase3_seed{N}_triggers.png             # Timeline: all triggers vs grokking
  phase3_seed{N}_kinematics.png           # Cv, velocity, acceleration plots
  phase4b_seed{N}_cv_adamw.png            # Cold-start: accuracy + Z + WD
  phase4b_wocs_seed{N}_cv_adamw.png       # Continuous: accuracy + Z + WD
```

## What to Compare

1. **Grokking epoch** across the 3 runs per seed:
   - Phase 3 (no intervention) → baseline
   - Phase 4B cold-start → how much faster?
   - Phase 4B continuous → how much faster?

2. **Trigger timing** in Phase 3:
   - When does `ema_zscore_continuous` fire vs `ema_zscore_gated`?
   - The delta between them shows the cold-start penalty

3. **Consistency across seeds**:
   - Does one variant have lower variance in grokking epoch?
   - Does the continuous sensor trigger more reliably?

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `z_thresh` | 2.0 | 2σ is a standard statistical anomaly threshold |
| `beta_z` | 0.9 | Effective window ≈ 10 epochs for EMA |
| `base_wd` | 0.1 | Standard regularization baseline |
| `train_acc gate` | 0.99 | Model has fully memorized → Cv dynamics become meaningful |
| `warmup` | 15 | Ignore initialization noise in first few epochs |
