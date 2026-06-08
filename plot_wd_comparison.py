"""
Weight Decay Comparison: Phase 4A vs Phase 4B
==============================================
Overlays the weight decay profile from both intervention strategies
on a single plot, highlighting discrete vs continuous heat injection.

Usage:
    python plot_wd_comparison.py --seed 42
    python plot_wd_comparison.py  (auto-detects seed from available files)
"""

import os
import csv
import glob
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150
})


def load_csv(filepath):
    data = {"epoch": [], "val_acc": [], "cv": [], "wd": []}
    if not os.path.exists(filepath):
        print(f"  Warning: {filepath} not found")
        return None
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["val_acc"].append(float(row["val_acc"]))
            data["cv"].append(float(row["cv"]))
            if "weight_decay" in row:
                data["wd"].append(float(row["weight_decay"]))
            elif "dynamic_wd" in row:
                data["wd"].append(float(row["dynamic_wd"]))
            else:
                data["wd"].append(0.1)
    return data


def find_seed(prefix):
    """Auto-detect seed from available data files."""
    pattern = f"data/{prefix}_seed*.csv"
    matches = glob.glob(pattern)
    if matches:
        # Extract seed from filename like phase4a_exp3_metrics_seed42.csv
        fname = os.path.basename(matches[0])
        seed_str = fname.split("_seed")[1].replace(".csv", "")
        return int(seed_str)
    # Fallback: try old naming without seed
    if os.path.exists(f"data/{prefix}.csv"):
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot WD comparison: 4A vs 4B")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to plot (default: auto-detect)")
    args = parser.parse_args()

    seed = args.seed

    # Try to find files
    if seed is not None:
        f4a = f"data/phase4a_exp3_metrics_seed{seed}.csv"
        f4b = f"data/phase4b_exp3_metrics_seed{seed}.csv"
    else:
        # Auto-detect
        seed = find_seed("phase4a_exp3_metrics")
        if seed is not None:
            f4a = f"data/phase4a_exp3_metrics_seed{seed}.csv"
            f4b = f"data/phase4b_exp3_metrics_seed{seed}.csv"
        else:
            # Fallback to old naming
            f4a = "data/phase4a_exp3_metrics.csv"
            f4b = "data/phase4b_exp3_metrics.csv"

    print(f"Loading Phase 4A: {f4a}")
    print(f"Loading Phase 4B: {f4b}")

    data_4a = load_csv(f4a)
    data_4b = load_csv(f4b)

    if data_4a is None and data_4b is None:
        print("No data files found. Run the experiments first.")
        return

    os.makedirs("figures", exist_ok=True)

    # --- Plot: Weight Decay comparison ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: WD over time for both
    if data_4a:
        ax1.plot(data_4a["epoch"], data_4a["wd"], color="#e74c3c", lw=2,
                 label="Phase 4A (Step-Function)", alpha=0.9)
    if data_4b:
        ax1.plot(data_4b["epoch"], data_4b["wd"], color="#2ecc71", lw=2,
                 label="Phase 4B (CvAdamW)", alpha=0.9)

    ax1.axhline(y=0.1, color="gray", ls=":", lw=1, alpha=0.7, label="Baseline WD (0.1)")
    ax1.set_ylabel("Weight Decay ($\\lambda_t$)")
    ax1.set_title("Weight Decay Profile: Discrete (4A) vs Continuous (4B)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Val accuracy for both (to show grokking timing)
    if data_4a:
        ax2.plot(data_4a["epoch"], data_4a["val_acc"], color="#e74c3c", lw=2,
                 label="Phase 4A Val Acc", alpha=0.9)
    if data_4b:
        ax2.plot(data_4b["epoch"], data_4b["val_acc"], color="#2ecc71", lw=2,
                 label="Phase 4B Val Acc", alpha=0.9)

    ax2.axhline(y=0.9, color="gray", ls="--", lw=1, alpha=0.5, label="Grokking threshold (0.9)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Generalization: When Does Each Method Grok?")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    seed_suffix = f"_seed{seed}" if seed else ""
    out_path = f"figures/wd_comparison_4a_vs_4b{seed_suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
