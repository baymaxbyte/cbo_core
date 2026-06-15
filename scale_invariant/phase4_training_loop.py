"""
CBO Phase 4A: Active Thermodynamic Optimization (Training Loop Method)
=======================================================================
Uses the Kinematic trigger from Phase 3 to detect the Cv peak, then
SPIKES weight decay to inject heat and force grokking immediately.

Mechanism:
    1. Train normally with base_wd=0.1
    2. Monitor Cv kinematics (velocity > 0, acceleration < 0 = peak crest)
    3. On trigger: spike weight_decay to 1.0-2.0 for a cooldown period
    4. After cooldown: return to base_wd=0.1

Physics:
    T_eff ∝ √d_k / ||W||²
    Spiking weight decay → shrinks ||W|| → increases T_eff → thermal shock
    Forces model out of memorization crater into generalization valley

Outputs:
    - data/phase4a_metrics.csv
    - data/phase4a_results.json (trigger epoch, grokking epoch, lag)
    - figures/phase4a_intervention.png

Usage:
    python phase4_training_loop.py --device cuda --epochs 4000
    python phase4_training_loop.py --device cuda --spike_wd 2.0 --cooldown 100
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
# RoPE + Model (same as Phase 2/3)
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
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.last_attn_logits = None

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv_proj(h).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        freqs = self.rope_freqs.to(x.device)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        self.last_attn_logits = attn_logits.detach()
        out = torch.matmul(F.softmax(attn_logits, dim=-1), v).transpose(1, 2).reshape(B, T, C)
        x = x + self.out_proj(out)
        return x + self.ff(self.ln2(x))


class ThermodynamicTransformer(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.p = p
        self.d_head = d_model // n_heads
        self.tok_emb = nn.Embedding(p + 3, d_model)
        rope_freqs = precompute_rope_freqs(self.d_head, max_seq_len=4)
        self.register_buffer("rope_freqs", rope_freqs)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, rope_freqs) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

    def forward(self, x):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_final(h)[:, -1, :])
        all_l = [b.last_attn_logits.reshape(-1) for b in self.blocks if b.last_attn_logits is not None]
        cv = torch.var(torch.cat(all_l)).item() if all_l else 0.0
        return logits, cv


# =============================================================================
# Kinematic Tracker
# =============================================================================

class KinematicTracker:
    """Detects the Cv peak crest using second-derivative analysis."""

    def __init__(self, warmup=15):
        self.warmup = warmup
        self.prev_cv = None
        self.prev_velocity = None
        self.triggered = False
        self.trigger_epoch = None

    def step(self, epoch, cv):
        if self.prev_cv is None:
            self.prev_cv = cv
            return False
        velocity = cv - self.prev_cv
        acceleration = velocity - self.prev_velocity if self.prev_velocity is not None else 0.0
        self.prev_velocity = velocity
        self.prev_cv = cv

        if not self.triggered and epoch > self.warmup:
            if velocity > 0 and acceleration < 0:
                self.triggered = True
                self.trigger_epoch = epoch
                return True
        return False


# =============================================================================
# Dataset
# =============================================================================

def create_dataset(p, seed=42):
    op_plus, op_equals = p, p + 1
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    targets = (a + b) % p
    inputs = torch.stack([a, torch.full_like(a, op_plus), b, torch.full_like(a, op_equals)], dim=1)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=gen)
    n = p * p // 2
    return (TensorDataset(inputs[perm[:n]], targets[perm[:n]]),
            TensorDataset(inputs[perm[n:]], targets[perm[n:]]))


# =============================================================================
# Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CBO Phase 4A: Training Loop Intervention")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--base_wd", type=float, default=0.1)
    parser.add_argument("--spike_wd", type=float, default=1.0)
    parser.add_argument("--cooldown", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto": device = "cpu"

    print(f"Phase 4A: Active Thermodynamic Optimization (Training Loop)")
    print(f"  base_wd={args.base_wd}, spike_wd={args.spike_wd}, cooldown={args.cooldown}")
    print(f"  p={args.p}, epochs={args.epochs}, device={device}\n")

    train_ds, val_ds = create_dataset(args.p, seed=42)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    model = ThermodynamicTransformer(p=args.p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=args.base_wd)
    criterion = nn.CrossEntropyLoss()
    tracker = KinematicTracker(warmup=args.warmup)

    is_heating = False
    trigger_epoch = -1
    rows = []
    wd_history = []

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = n_batches = 0
        epoch_cv = 0.0
        cv_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, cv = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            epoch_cv += cv
            cv_count += 1

        avg_cv = epoch_cv / max(cv_count, 1)
        avg_loss = epoch_loss / max(n_batches, 1)

        # Check kinematic trigger
        if not is_heating and trigger_epoch == -1:
            peak_detected = tracker.step(epoch, avg_cv)
            if peak_detected:
                print(f"  🔥 KINEMATIC TRIGGER at epoch {epoch}! Injecting heat (wd={args.spike_wd})")
                is_heating = True
                trigger_epoch = epoch
                for pg in optimizer.param_groups:
                    pg['weight_decay'] = args.spike_wd

        # Cooldown
        if is_heating and epoch == trigger_epoch + args.cooldown:
            print(f"  ❄️ Cooldown at epoch {epoch}. Returning to wd={args.base_wd}")
            is_heating = False
            for pg in optimizer.param_groups:
                pg['weight_decay'] = args.base_wd

        current_wd = optimizer.param_groups[0]['weight_decay']
        wd_history.append(current_wd)

        # Validation
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
            val_acc = correct / total

            correct = total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
            train_acc = correct / total

        rows.append({"epoch": epoch+1, "train_loss": avg_loss, "train_acc": train_acc,
                     "val_acc": val_acc, "cv": avg_cv, "weight_decay": current_wd})

        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Loss:{avg_loss:.4f} TrAcc:{train_acc:.4f} "
                  f"VAcc:{val_acc:.4f} Cv:{avg_cv:.4f} WD:{current_wd}")

    # Detect grokking
    grok_epoch = None
    for r in rows:
        if r["val_acc"] > 0.9:
            grok_epoch = r["epoch"]
            break

    # Save
    with open("data/phase4a_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    results = {"trigger_epoch": trigger_epoch, "grokking_epoch": grok_epoch,
               "cooldown": args.cooldown, "spike_wd": args.spike_wd,
               "lag": (grok_epoch - trigger_epoch) if grok_epoch and trigger_epoch >= 0 else None}
    with open("data/phase4a_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Trigger: epoch {trigger_epoch}")
    print(f"  Grokking: epoch {grok_epoch}")
    print(f"  Lag: {results['lag']} epochs")
    print(f"{'='*50}")

    # Plot
    epochs_arr = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs_arr, [r["val_acc"] for r in rows], "b-", lw=1.5, label="Val Acc")
    ax1.plot(epochs_arr, [r["train_acc"] for r in rows], "r--", lw=1, alpha=0.5, label="Train Acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    cv_vals = [r["cv"] for r in rows]
    cv_max = max(cv_vals) if max(cv_vals) > 0 else 1
    ax2.plot(epochs_arr, [c/cv_max for c in cv_vals], "g-", lw=1, alpha=0.6, label="Cv (norm)")
    ax2.fill_between(epochs_arr, 0, [r["weight_decay"]/args.spike_wd for r in rows],
                     alpha=0.15, color="orange", label="WD active")
    ax2.set_ylabel("Cv / WD")
    if trigger_epoch >= 0:
        ax1.axvline(x=trigger_epoch, color="red", ls="--", lw=2, label=f"Trigger: {trigger_epoch}")
    if grok_epoch:
        ax1.axvline(x=grok_epoch, color="black", ls="-", lw=2, label=f"Grok: {grok_epoch}")
    fig.suptitle("Phase 4A: Active Heat Injection via Training Loop", fontsize=13)
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    fig.legend(lines1+lines2, l1+l2, loc="lower center", ncol=4, bbox_to_anchor=(0.5,-0.03), fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("figures/phase4a_intervention.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/phase4a_intervention.png")


if __name__ == "__main__":
    main()
