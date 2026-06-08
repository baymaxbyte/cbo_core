"""
CBO Phase 3: Multi-Method Trigger Benchmarking
================================================
Trains the baseline model normally (same as Phase 2) but simultaneously
runs three "ghost" trigger mechanisms that log WHEN they would have fired,
without affecting the optimizer.

Trigger Methods:
    1. Profiler (Static Threshold): Fires when dCv/dt > tau
    2. Z-Score (Statistical Anomaly): Fires when velocity z-score > threshold
    3. Kinematic (Second Derivative): Fires when velocity > 0 AND acceleration < 0
       (the exact mathematical crest of the Cv peak)

Purpose:
    Proves that the trigger mechanisms can anticipate the phase transition
    BEFORE natural grokking occurs. The "lag" between trigger and grokking
    is what Phase 4 will eliminate.

Outputs:
    - data/phase3_metrics.csv (epoch, train_loss, val_loss, train_acc, val_acc, cv)
    - data/phase3_benchmark_results.json (trigger epochs, physics logs)
    - figures/phase3_triggers.png (timeline showing all triggers vs grokking)
    - figures/phase3_kinematics.png (cv, velocity, acceleration plots)

Usage:
    python phase3_benchmark.py --device cuda --epochs 4000
    python phase3_benchmark.py --device cuda --p 37 --epochs 2000  # quick test
"""

import argparse
import os
import math
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# RoPE
# =============================================================================

def precompute_rope_freqs(d_head, max_seq_len=16, theta_base=10000.0):
    freqs = 1.0 / (theta_base ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, freqs_cis):
    B, H, T, D = x.shape
    x_pairs = x.float().reshape(B, H, T, D // 2, 2)
    x_complex = torch.view_as_complex(x_pairs)
    freqs = freqs_cis[:T].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape(B, H, T, D).type_as(x)


# =============================================================================
# Model (same as Phase 2)
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, rope_freqs):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope_freqs = rope_freqs
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.last_attn_logits = None

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv_proj(h).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        freqs = self.rope_freqs.to(x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        scale = math.sqrt(self.d_head)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale
        self.last_attn_logits = attn_logits.detach()
        attn = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        x = x + self.out_proj(out)
        x = x + self.ff(self.ln2(x))
        return x


class ThermodynamicTransformer(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.p = p
        self.d_head = d_model // n_heads
        self.seq_len = 4
        self.tok_emb = nn.Embedding(p + 3, d_model)
        rope_freqs = precompute_rope_freqs(self.d_head, max_seq_len=self.seq_len)
        self.register_buffer("rope_freqs", rope_freqs)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, rope_freqs) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

    def forward(self, x, compute_cv=True):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_final(h)
        logits = self.head(h[:, -1, :])
        cv = None
        if compute_cv:
            all_logits = []
            for block in self.blocks:
                if block.last_attn_logits is not None:
                    all_logits.append(block.last_attn_logits.reshape(-1))
            if all_logits:
                cv = torch.var(torch.cat(all_logits)).item()
        return logits, cv


# =============================================================================
# Thermodynamic Benchmarker (Multi-Method Ghost Triggers)
# =============================================================================

class ThermodynamicBenchmarker:
    """
    Evaluates three trigger mechanisms simultaneously without affecting training.
    Logs the exact epoch each method would have fired.
    Uses EMA to smooth the Cv signal and ignore thermal micro-ripples.
    """

    def __init__(self, static_tau=0.015, z_score_threshold=3.0, warmup_epochs=15, alpha=0.9):
        self.warmup_epochs = warmup_epochs
        self.static_tau = static_tau
        self.z_threshold = z_score_threshold
        self.alpha = alpha  # EMA smoothing factor

        # Physics state
        self.ema_cv = None  # Smoothed Cv
        self.prev_ema_cv = None
        self.prev_velocity = None
        self.velocity_history = []

        # Trigger logs (stores the epoch each method fired)
        self.triggers = {
            "profiler": None,
            "z_score": None,
            "kinematic": None,
        }

        # Continuous data logs
        self.logs = {
            "cv": [],
            "velocity": [],
            "acceleration": [],
        }

    def step(self, epoch, current_cv, train_accuracy=0.0):
        """Process one epoch's Cv value through all trigger mechanisms.
        Uses EMA smoothing + Memorization Gate."""

        # Handle initialization
        if self.ema_cv is None:
            self.ema_cv = current_cv
            self.prev_ema_cv = current_cv
            self.logs["cv"].append(current_cv)
            self.logs["velocity"].append(0.0)
            self.logs["acceleration"].append(0.0)
            return

        # 1. Apply EMA Smoothing to the raw signal
        self.ema_cv = (self.alpha * self.ema_cv) + ((1 - self.alpha) * current_cv)

        # 2. Calculate kinematics on the SMOOTHED signal
        velocity = self.ema_cv - self.prev_ema_cv
        acceleration = velocity - self.prev_velocity if self.prev_velocity is not None else 0.0

        # Log continuous data
        self.logs["cv"].append(self.ema_cv)
        self.logs["velocity"].append(velocity)
        self.logs["acceleration"].append(acceleration)

        # 3. THE MEMORIZATION GATE:
        # Only allow triggers to evaluate if model has memorized AND past warmup
        if epoch > self.warmup_epochs and train_accuracy >= 0.99:

            # --- METHOD 1: PROFILER (Static Threshold) ---
            if self.triggers["profiler"] is None and velocity > self.static_tau:
                self.triggers["profiler"] = epoch
                print(f"    [PROFILER] Triggered at epoch {epoch} (velocity={velocity:.6f} > tau={self.static_tau})")

            # --- METHOD 2: Z-SCORE (Statistical Anomaly) ---
            if self.triggers["z_score"] is None and len(self.velocity_history) > 5:
                mean_v = np.mean(self.velocity_history)
                std_v = np.std(self.velocity_history) + 1e-8
                z_score = (velocity - mean_v) / std_v
                if z_score > self.z_threshold:
                    self.triggers["z_score"] = epoch
                    print(f"    [Z-SCORE] Triggered at epoch {epoch} (z={z_score:.2f} > {self.z_threshold})")

            # --- METHOD 3: KINEMATIC (Second Derivative) ---
            # Velocity is positive (climbing) but acceleration is negative (cresting)
            if self.triggers["kinematic"] is None:
                if velocity > 0 and acceleration < 0:
                    self.triggers["kinematic"] = epoch
                    print(f"    [KINEMATIC] Triggered at epoch {epoch} (v={velocity:.6f}>0, a={acceleration:.6f}<0)")

        # Update state history
        if self.triggers["z_score"] is None:
            self.velocity_history.append(velocity)

        self.prev_velocity = velocity
        self.prev_ema_cv = self.ema_cv


# =============================================================================
# Dataset
# =============================================================================

def create_dataset(p, seed=42):
    op_plus = p
    op_equals = p + 1
    a_vals = torch.arange(p).repeat_interleave(p)
    b_vals = torch.arange(p).repeat(p)
    targets = (a_vals + b_vals) % p
    inputs = torch.stack([
        a_vals, torch.full_like(a_vals, op_plus),
        b_vals, torch.full_like(a_vals, op_equals),
    ], dim=1)
    n_total = p * p
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    n_train = n_total // 2
    train_ds = TensorDataset(inputs[perm[:n_train]], targets[perm[:n_train]])
    val_ds = TensorDataset(inputs[perm[n_train:]], targets[perm[n_train:]])
    return train_ds, val_ds


# =============================================================================
# Training
# =============================================================================

def train_phase3(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed handling: generate random seed if not provided
    if args.seed is not None:
        seed = args.seed
    else:
        seed = torch.randint(0, 2**31, (1,)).item()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    p = args.p
    print(f"Phase 3: Multi-Method Trigger Benchmarking")
    print(f"  p={p}, epochs={args.epochs}, seed={seed}")
    print(f"  Profiler tau={args.tau}, Z-score threshold={args.z_threshold}")
    print(f"  Device: {device}")
    print()

    # Dataset
    train_ds, val_ds = create_dataset(p, seed=42)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    # Model
    model = ThermodynamicTransformer(p=p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    # Benchmarker
    benchmarker = ThermodynamicBenchmarker(
        static_tau=args.tau,
        z_score_threshold=args.z_threshold,
        warmup_epochs=args.warmup,
    )

    # Logging
    rows = []
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    train_acc = 0.0  # Initialize before loop (memorization gate starts closed)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_cv_sum = 0.0
        epoch_cv_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, cv = model(x, compute_cv=True)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if cv is not None:
                epoch_cv_sum += cv
                epoch_cv_count += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_cv = epoch_cv_sum / max(epoch_cv_count, 1)

        # Feed Cv to benchmarker (ghost triggers) WITH MEMORIZATION GATE
        benchmarker.step(epoch, avg_cv, train_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            correct = total = 0
            val_loss_sum = 0.0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x, compute_cv=False)
                val_loss_sum += criterion(logits, y).item() * y.size(0)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
            val_loss = val_loss_sum / total
            val_acc = correct / total

            correct = total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x, compute_cv=False)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
            train_acc = correct / total

        rows.append({
            "epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc, "cv": avg_cv,
        })

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"TrLoss:{avg_loss:.4f} VLoss:{val_loss:.4f} | "
                  f"TrAcc:{train_acc:.4f} VAcc:{val_acc:.4f} | Cv:{avg_cv:.4f}")

    # Detect grokking epoch (first time val_acc > 0.9)
    grok_epoch = None
    for r in rows:
        if r["val_acc"] > 0.9:
            grok_epoch = r["epoch"]
            break

    # Save results
    csv_path = f"data/phase3_exp3_metrics_seed{seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                               "train_acc", "val_acc", "cv"])
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "seed": seed,
        "trigger_epochs": benchmarker.triggers,
        "grokking_epoch": grok_epoch,
        "physics_logs": benchmarker.logs,
        "config": {"p": p, "epochs": args.epochs, "tau": args.tau,
                   "z_threshold": args.z_threshold, "warmup": args.warmup,
                   "seed": seed},
    }
    with open(f"data/phase3_exp3_benchmark_results_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"PHASE 3 RESULTS (seed={seed})")
    print(f"{'='*50}")
    print(f"  Profiler triggered:  epoch {benchmarker.triggers['profiler']}")
    print(f"  Z-Score triggered:   epoch {benchmarker.triggers['z_score']}")
    print(f"  Kinematic triggered: epoch {benchmarker.triggers['kinematic']}")
    print(f"  Natural grokking:    epoch {grok_epoch}")
    if grok_epoch and benchmarker.triggers["kinematic"]:
        lag = grok_epoch - benchmarker.triggers["kinematic"]
        print(f"  LAG (kinematic -> grokking): {lag} epochs")
    print(f"{'='*50}")

    # Plot
    plot_phase3(rows, benchmarker, grok_epoch, seed)
    print(f"\n  Saved: data/phase3_exp3_metrics_seed{seed}.csv")
    print(f"  Saved: data/phase3_exp3_benchmark_results_seed{seed}.json")
    print(f"  Saved: figures/phase3_exp3_triggers_seed{seed}.png")
    print(f"  Saved: figures/phase3_exp3_kinematics_seed{seed}.png")


# =============================================================================
# Plotting
# =============================================================================

def plot_phase3(rows, benchmarker, grok_epoch, seed):
    epochs = [r["epoch"] for r in rows]
    val_acc = [r["val_acc"] for r in rows]
    cv = benchmarker.logs["cv"]
    velocity = benchmarker.logs["velocity"]
    acceleration = benchmarker.logs["acceleration"]

    # Pad if needed
    while len(cv) < len(epochs):
        cv.append(cv[-1] if cv else 0)
    while len(velocity) < len(epochs):
        velocity.append(0)
    while len(acceleration) < len(epochs):
        acceleration.append(0)

    # Figure 1: Trigger Timeline
    fig, ax1 = plt.subplots(figsize=(12, 6))
    cv_max = max(cv) if max(cv) > 0 else 1.0
    ax1.plot(epochs[:len(cv)], [c / cv_max for c in cv], "r-", lw=1.5, label="$C_v$ (normalized)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("$C_v$ (normalized)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, "b--", lw=1.5, label="Val Accuracy")
    ax2.set_ylabel("Validation Accuracy", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(-0.05, 1.05)

    # Mark triggers
    colors = {"profiler": "orange", "z_score": "purple", "kinematic": "green"}
    for method, ep in benchmarker.triggers.items():
        if ep is not None:
            ax1.axvline(x=ep, color=colors[method], ls="--", lw=2, alpha=0.8,
                        label=f"{method}: epoch {ep}")
    if grok_epoch:
        ax1.axvline(x=grok_epoch, color="black", ls="-", lw=2, alpha=0.8,
                    label=f"Grokking: epoch {grok_epoch}")

    fig.suptitle("Phase 3: Trigger Benchmarking — When Would Each Method Fire?", fontsize=13)
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, l1 + l2, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"figures/phase3_exp3_triggers_seed{seed}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Kinematics (Cv, velocity, acceleration)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(epochs[:len(cv)], cv, "r-", lw=1)
    axes[0].set_ylabel("$C_v$ (raw)")
    axes[0].set_title("Specific Heat")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs[:len(velocity)], velocity, "g-", lw=1)
    axes[1].axhline(y=0, color="gray", ls=":", alpha=0.5)
    axes[1].set_ylabel("$v = dC_v/dt$")
    axes[1].set_title("Velocity (First Derivative)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs[:len(acceleration)], acceleration, "m-", lw=1)
    axes[2].axhline(y=0, color="gray", ls=":", alpha=0.5)
    axes[2].set_ylabel("$a = d^2C_v/dt^2$")
    axes[2].set_title("Acceleration (Second Derivative)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    # Mark triggers on all subplots
    for method, ep in benchmarker.triggers.items():
        if ep is not None:
            for ax in axes:
                ax.axvline(x=ep, color=colors[method], ls="--", lw=1.5, alpha=0.6)

    fig.suptitle("Phase 3 (Exp 3 - EMA Smoothed): Thermodynamic Kinematics", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"figures/phase3_exp3_kinematics_seed{seed}.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBO Phase 3: Trigger Benchmarking")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--tau", type=float, default=0.015,
                        help="Static threshold for profiler trigger")
    parser.add_argument("--z_threshold", type=float, default=3.0,
                        help="Z-score threshold for statistical trigger")
    parser.add_argument("--warmup", type=int, default=15,
                        help="Warmup epochs before triggers can fire")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")
    args = parser.parse_args()
    train_phase3(args)
