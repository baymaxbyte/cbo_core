"""
CBO Phase 4B: Scale-Invariant CvAdamW — Z-Score Driven Optimizer
==================================================================
A custom PyTorch optimizer that dynamically scales weight decay based on
the Z-Score of the thermodynamic velocity (dCv/dt).

Instead of relying on absolute magnitudes (kappa, tau), we treat the
thermodynamic velocity v_t as a random variable and track its distribution
dynamically using EMA-based rolling statistics.

Formula:
    v_t = Cv(t) - Cv(t-1)
    mu_t = beta_z * mu_{t-1} + (1 - beta_z) * v_t
    sigma_t^2 = beta_z * sigma_{t-1}^2 + (1 - beta_z) * (v_t - mu_{t-1}) * (v_t - mu_t)
    Z_t = (v_t - mu_t) / (sqrt(sigma_t^2) + eps)
    lambda_t = base_wd + max(0, Z_t - z_thresh)

This is scale-invariant: no arbitrary kappa or tau needed. Only a universal
statistical constant z_thresh (e.g., 2.0 standard deviations) identifies
a true phase transition.

Usage:
    python phase4_cv_adamw.py --device cuda --epochs 7000 --seed 42
    python phase4_cv_adamw.py --device cuda --z_thresh 2.0 --beta_z 0.9
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
# Scale-Invariant CvAdamW: Z-Score Driven Optimizer
# =============================================================================

class CvAdamW(torch.optim.Optimizer):
    """
    AdamW with dynamic weight decay driven by Z-Score of Cv velocity.

    The weight decay is continuously modulated:
        lambda_t = base_wd + max(0, Z_t - z_thresh)

    Where Z_t is the z-score of current velocity relative to its
    EMA-tracked distribution. This is fully scale-invariant.

    Args:
        params: model parameters
        lr: learning rate (default: 3e-4)
        betas: Adam momentum coefficients (default: (0.9, 0.999))
        eps: numerical stability (default: 1e-8)
        base_weight_decay: baseline weight decay (default: 0.1)
        z_thresh: z-score threshold for activation (default: 2.0)
        beta_z: EMA smoothing for velocity statistics (default: 0.9)
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 base_weight_decay=0.1, z_thresh=2.0, beta_z=0.9):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=base_weight_decay,
                        z_thresh=z_thresh, beta_z=beta_z)
        super(CvAdamW, self).__init__(params, defaults)

        # Optimizer-level thermodynamic state
        self._thermo_state = {
            'prev_cv': None,
            'mu': 0.0,           # EMA of velocity
            'sigma_sq': 0.0,     # EMA of variance
            'z_score': 0.0,      # Current z-score
            'velocity': 0.0,     # Current velocity
            'dynamic_wd': base_weight_decay,
            'step_count': 0,
        }

    def get_dynamic_wd(self):
        """Return the current dynamic weight decay value."""
        return self._thermo_state['dynamic_wd']

    def get_z_score(self):
        """Return the current z-score."""
        return self._thermo_state['z_score']

    def get_velocity(self):
        """Return the current velocity."""
        return self._thermo_state['velocity']

    def get_mu(self):
        """Return the current EMA mean of velocity."""
        return self._thermo_state['mu']

    def get_sigma(self):
        """Return the current standard deviation."""
        return math.sqrt(self._thermo_state['sigma_sq'] + 1e-10)

    @torch.no_grad()
    def step(self, current_cv=None, train_accuracy=0.0, closure=None):
        """
        Performs a single optimization step with z-score driven weight decay.

        Args:
            current_cv: the current specific heat value (float)
            train_accuracy: current training accuracy (used as memorization gate)
            closure: optional closure for loss computation
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Update thermodynamic state
        base_wd = self.defaults['weight_decay']
        z_thresh = self.defaults['z_thresh']
        beta_z = self.defaults['beta_z']
        eps = 1e-10

        if current_cv is not None:
            ts = self._thermo_state
            ts['step_count'] += 1

            if ts['prev_cv'] is None:
                ts['prev_cv'] = current_cv
                v_t = 0.0
            else:
                v_t = current_cv - ts['prev_cv']

            ts['velocity'] = v_t

            # THE MEMORIZATION GATE: Only track stats once the model is stable
            if train_accuracy >= 0.99:
                # EMA of velocity (rolling mean)
                mu_prev = ts['mu']
                mu_t = beta_z * mu_prev + (1 - beta_z) * v_t
                ts['mu'] = mu_t

                # EMA of variance (Welford-style with EMA)
                sigma_sq_t = beta_z * ts['sigma_sq'] + (1 - beta_z) * (v_t - mu_prev) * (v_t - mu_t)
                ts['sigma_sq'] = max(0.0, sigma_sq_t)

                # Z-Score
                sigma_t = math.sqrt(ts['sigma_sq'] + eps)
                z_t = (v_t - mu_t) / sigma_t
                ts['z_score'] = z_t

                # Dynamic weight decay
                ts['dynamic_wd'] = base_wd + max(0.0, z_t - z_thresh)
            else:
                # Keep tracking baseline, but do not update Z-score logic
                ts['z_score'] = 0.0
                ts['dynamic_wd'] = base_wd

            # ALWAYS update prev_cv silently
            ts['prev_cv'] = current_cv
        else:
            dynamic_wd = self._thermo_state['dynamic_wd']

        # 2. Standard AdamW update with dynamic weight decay
        dynamic_wd = self._thermo_state['dynamic_wd']
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps_adam = group['eps']

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
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps_adam)

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
    parser = argparse.ArgumentParser(description="CBO Phase 4B: Scale-Invariant CvAdamW (Z-Score)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--z_thresh", type=float, default=2.0, help="Z-score threshold")
    parser.add_argument("--beta_z", type=float, default=0.9, help="EMA smoothing for velocity stats")
    parser.add_argument("--base_wd", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto": device = "cpu"

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"Phase 4B: Scale-Invariant CvAdamW (Z-Score Driven)")
    print(f"  z_thresh={args.z_thresh}, beta_z={args.beta_z}, base_wd={args.base_wd}")
    print(f"  p={args.p}, epochs={args.epochs}, seed={args.seed}, device={device}")
    print(f"  Formula: lambda_t = base_wd + max(0, Z_t - z_thresh)")
    print()

    train_ds, val_ds = create_dataset(args.p, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    model = ThermodynamicTransformer(p=args.p).to(device)
    optimizer = CvAdamW(model.parameters(), lr=3e-4,
                        base_weight_decay=args.base_wd,
                        z_thresh=args.z_thresh, beta_z=args.beta_z)
    criterion = nn.CrossEntropyLoss()

    rows = []
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    prev_train_acc = 0.0

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
        z_score = optimizer.get_z_score()
        velocity = optimizer.get_velocity()
        mu = optimizer.get_mu()
        sigma = optimizer.get_sigma()

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

        prev_train_acc = train_acc  # Update for next epoch's gate

        rows.append({"epoch": epoch+1, "train_loss": avg_loss, "train_acc": train_acc,
                     "val_acc": val_acc, "cv": avg_cv, "velocity": velocity,
                     "mu": mu, "sigma": sigma, "z_score": z_score,
                     "dynamic_wd": dynamic_wd})

        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss:{avg_loss:.4f} TrAcc:{train_acc:.4f} "
                  f"VAcc:{val_acc:.4f} | Cv:{avg_cv:.6f} v:{velocity:.6f} "
                  f"mu:{mu:.6f} sigma:{sigma:.6f} Z:{z_score:.3f} WD:{dynamic_wd:.4f}")

    # Detect grokking
    grok_epoch = None
    for r in rows:
        if r["val_acc"] > 0.9:
            grok_epoch = r["epoch"]
            break

    # Save
    csv_path = f"data/phase4b_seed{args.seed}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    results = {"grokking_epoch": grok_epoch, "z_thresh": args.z_thresh,
               "beta_z": args.beta_z, "base_wd": args.base_wd, "seed": args.seed,
               "max_dynamic_wd": max(r["dynamic_wd"] for r in rows),
               "max_z_score": max(r["z_score"] for r in rows)}
    json_path = f"data/phase4b_seed{args.seed}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  PHASE 4B RESULTS (seed={args.seed})")
    print(f"{'='*60}")
    print(f"  Grokking epoch:  {grok_epoch}")
    print(f"  Max Z-score:     {results['max_z_score']:.4f}")
    print(f"  Max dynamic WD:  {results['max_dynamic_wd']:.4f}")
    print(f"  z_thresh:        {args.z_thresh}")
    print(f"  beta_z:          {args.beta_z}")
    print(f"{'='*60}")

    # Plot
    epochs_arr = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: Accuracy
    axes[0].plot(epochs_arr, [r["val_acc"] for r in rows], "b-", lw=1.5, label="Val Acc")
    axes[0].plot(epochs_arr, [r["train_acc"] for r in rows], "r--", lw=1, alpha=0.5, label="Train Acc")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"Phase 4B: Scale-Invariant CvAdamW (seed={args.seed}, z_thresh={args.z_thresh})")
    axes[0].grid(True, alpha=0.3)
    if grok_epoch:
        axes[0].axvline(x=grok_epoch, color="black", ls="-", lw=2, alpha=0.7, label=f"Grok: {grok_epoch}")
        axes[0].legend(loc="upper left")

    # Panel 2: Z-Score and WD
    ax2a = axes[1]
    ax2b = ax2a.twinx()
    ax2a.plot(epochs_arr, [r["z_score"] for r in rows], "purple", lw=1, label="Z-score")
    ax2a.axhline(y=args.z_thresh, color="red", ls="--", lw=1.5, alpha=0.7, label=f"z_thresh={args.z_thresh}")
    ax2a.set_ylabel("Z-score", color="purple")
    ax2b.plot(epochs_arr, [r["dynamic_wd"] for r in rows], "orange", lw=1.5, label="Dynamic WD")
    ax2b.set_ylabel("Weight Decay", color="orange")
    ax2a.legend(loc="upper left")
    ax2b.legend(loc="upper right")
    ax2a.grid(True, alpha=0.3)

    # Panel 3: Cv and velocity
    ax3a = axes[2]
    ax3b = ax3a.twinx()
    ax3a.plot(epochs_arr, [r["cv"] for r in rows], "r-", lw=1, label="Cv")
    ax3a.set_ylabel("Cv", color="red")
    ax3b.plot(epochs_arr, [r["velocity"] for r in rows], "g-", lw=0.8, alpha=0.7, label="Velocity")
    ax3b.axhline(y=0, color="gray", ls=":", alpha=0.5)
    ax3b.set_ylabel("Velocity (dCv/dt)", color="green")
    ax3a.set_xlabel("Epoch")
    ax3a.legend(loc="upper left")
    ax3b.legend(loc="upper right")
    ax3a.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"figures/phase4b_seed{args.seed}_cv_adamw.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")
    print(f"  Saved: {fig_path}")


if __name__ == "__main__":
    main()
