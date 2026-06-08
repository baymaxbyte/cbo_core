"""
CBO Phase 4B: CvAdamW — Thermodynamically-Aware Custom Optimizer
==================================================================
A custom PyTorch optimizer that dynamically scales weight decay based on
the specific heat (Cv) momentum. No external tracker needed — all
thermodynamic logic lives inside the optimizer's .step() function.

Formula:
    lambda_t = base_wd + kappa * max(0, mu_t - tau)

Where:
    mu_t = EMA of dCv/dt (momentum of Cv velocity)
    tau = threshold below which no heat is injected
    kappa = amplification factor for heat injection

This provides CONTINUOUS, proportional heat injection rather than the
binary spike/cooldown of Phase 4A.

Usage:
    python phase4_cv_adamw.py --device cuda --epochs 4000
    python phase4_cv_adamw.py --device cuda --kappa 10.0 --tau 0.01
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
# CvAdamW: Thermodynamically-Aware Optimizer
# =============================================================================

class CvAdamW(torch.optim.Optimizer):
    """
    AdamW with dynamic weight decay driven by specific heat momentum.

    The weight decay is continuously modulated:
        lambda_t = base_wd + kappa * max(0, mu_t - tau)

    Where mu_t is the EMA of dCv/dt. When Cv is rising (approaching the
    phase transition), weight decay increases proportionally, injecting
    heat to accelerate grokking.

    Args:
        params: model parameters
        lr: learning rate (default: 3e-4)
        betas: Adam momentum coefficients (default: (0.9, 0.999))
        eps: numerical stability (default: 1e-8)
        base_weight_decay: baseline weight decay (default: 0.1)
        kappa: heat amplification factor (default: 5.0)
        tau: Cv velocity threshold for activation (default: 0.02)
        alpha: EMA smoothing for Cv momentum (default: 0.9)
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 base_weight_decay=0.1, kappa=5.0, tau=0.02, alpha=0.9):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=base_weight_decay,
                        kappa=kappa, tau=tau, alpha=alpha)
        super(CvAdamW, self).__init__(params, defaults)

        # Optimizer-level thermodynamic state
        self._thermo_state = {
            'prev_cv': None,
            'momentum_cv': 0.0,
            'dynamic_wd': base_weight_decay,
            'step_count': 0,
        }

    def get_dynamic_wd(self):
        """Return the current dynamic weight decay value."""
        return self._thermo_state['dynamic_wd']

    @torch.no_grad()
    def step(self, current_cv=None, train_accuracy=0.0, closure=None):
        """
        Performs a single optimization step with thermodynamic weight decay.
        The Memorization Gate: only modulates weight decay when train_accuracy >= 0.99.

        Args:
            current_cv: the current specific heat value (float)
            train_accuracy: current training accuracy (gate condition)
            closure: optional closure for loss computation
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Update thermodynamic state (ONLY if memorization gate is open)
        base_wd = self.defaults['weight_decay']
        kappa = self.defaults['kappa']
        tau = self.defaults['tau']
        alpha = self.defaults['alpha']

        if current_cv is not None:
            ts = self._thermo_state
            ts['step_count'] += 1

            if ts['prev_cv'] is None:
                ts['prev_cv'] = current_cv
                delta_cv = 0.0
            else:
                delta_cv = current_cv - ts['prev_cv']

            # Memorization Gate: only update momentum when model has memorized
            if train_accuracy >= 0.99:
                # EMA momentum of Cv velocity
                ts['momentum_cv'] = alpha * ts['momentum_cv'] + (1 - alpha) * delta_cv

                # Dynamic weight decay formula
                mu_t = ts['momentum_cv']
                dynamic_wd = base_wd + kappa * max(0.0, mu_t - tau)
                ts['dynamic_wd'] = dynamic_wd
            # else: momentum freezes (mu_t <- mu_{t-1})

            # prev_cv updated INSIDE the gate (the original bug)
            ts['prev_cv'] = current_cv

        # Fallback: use stored dynamic_wd if not computed this step
        dynamic_wd = self._thermo_state.get('dynamic_wd', base_wd)

        # 2. Standard AdamW update with dynamic weight decay
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decoupled weight decay (AdamW style) with DYNAMIC lambda
                p.data.mul_(1.0 - lr * dynamic_wd)

                # Adam moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# =============================================================================
# RoPE + Model (same architecture)
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
    parser = argparse.ArgumentParser(description="CBO Phase 4B: CvAdamW Optimizer")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--kappa", type=float, default=5.0, help="Heat amplification factor")
    parser.add_argument("--tau", type=float, default=0.02, help="Cv velocity threshold")
    parser.add_argument("--alpha", type=float, default=0.9, help="EMA smoothing")
    parser.add_argument("--base_wd", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto": device = "cpu"

    print(f"Phase 4B: CvAdamW — Thermodynamically-Aware Optimizer")
    print(f"  kappa={args.kappa}, tau={args.tau}, alpha={args.alpha}, base_wd={args.base_wd}")
    print(f"  p={args.p}, epochs={args.epochs}, device={device}\n")

    train_ds, val_ds = create_dataset(args.p, seed=42)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    model = ThermodynamicTransformer(p=args.p).to(device)
    optimizer = CvAdamW(model.parameters(), lr=3e-4,
                        base_weight_decay=args.base_wd,
                        kappa=args.kappa, tau=args.tau, alpha=args.alpha)
    criterion = nn.CrossEntropyLoss()

    rows = []
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    prev_train_acc = 0.0  # Track previous epoch's train accuracy for memorization gate

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
            # Pass Cv AND train_accuracy directly into the optimizer step!
            optimizer.step(current_cv=cv, train_accuracy=prev_train_acc)
            epoch_loss += loss.item()
            n_batches += 1
            epoch_cv += cv
            cv_count += 1

        avg_cv = epoch_cv / max(cv_count, 1)
        avg_loss = epoch_loss / max(n_batches, 1)
        dynamic_wd = optimizer.get_dynamic_wd()

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
                     "val_acc": val_acc, "cv": avg_cv, "dynamic_wd": dynamic_wd})
        prev_train_acc = train_acc  # Update for next epoch's memorization gate

        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Loss:{avg_loss:.4f} TrAcc:{train_acc:.4f} "
                  f"VAcc:{val_acc:.4f} Cv:{avg_cv:.4f} WD:{dynamic_wd:.4f}")

    # Detect grokking
    grok_epoch = None
    for r in rows:
        if r["val_acc"] > 0.9:
            grok_epoch = r["epoch"]
            break

    # Save
    with open("data/phase4b_exp3_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    results = {"grokking_epoch": grok_epoch, "kappa": args.kappa, "tau": args.tau,
               "alpha": args.alpha, "base_wd": args.base_wd,
               "max_dynamic_wd": max(r["dynamic_wd"] for r in rows)}
    with open("data/phase4b_exp3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Grokking: epoch {grok_epoch}")
    print(f"  Max dynamic WD: {results['max_dynamic_wd']:.4f}")
    print(f"{'='*50}")

    # Plot
    epochs_arr = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs_arr, [r["val_acc"] for r in rows], "b-", lw=1.5, label="Val Acc")
    ax1.plot(epochs_arr, [r["train_acc"] for r in rows], "r--", lw=1, alpha=0.5, label="Train Acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(epochs_arr, [r["dynamic_wd"] for r in rows], "orange", lw=1.5, label="Dynamic WD")
    cv_vals = [r["cv"] for r in rows]
    cv_max = max(cv_vals) if max(cv_vals) > 0 else 1
    ax2.plot(epochs_arr, [c/cv_max for c in cv_vals], "g-", lw=1, alpha=0.5, label="Cv (norm)")
    ax2.set_ylabel("WD / Cv")
    fig.suptitle(f"Phase 4B: CvAdamW (κ={args.kappa}, τ={args.tau})", fontsize=13)
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    fig.legend(lines1+lines2, l1+l2, loc="lower center", ncol=4, bbox_to_anchor=(0.5,-0.03), fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("figures/phase4b_exp3_cv_adamw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/phase4b_cv_adamw.png")


if __name__ == "__main__":
    main()
