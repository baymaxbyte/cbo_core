"""
Master Publication Figure: 3-Panel Comparison
==============================================
Creates a Nature/NeurIPS-style 3-panel vertical stack comparing:
- Panel A: Phase 3 (Baseline) — model trapped in memorization
- Panel B: Phase 4A (Step-Function) — forced grokking via localized heat
- Panel C: Phase 4B (CvAdamW) — continuous thermodynamic adjustment

Usage:
    python plot_master_figure.py
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})


def load_data(filepath):
    data = {"epoch": [], "train_acc": [], "val_acc": [], "cv": [], "wd": []}
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return data
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_acc"].append(float(row["train_acc"]))
            data["val_acc"].append(float(row["val_acc"]))
            data["cv"].append(float(row["cv"]))
            # Handle different weight decay keys
            if "weight_decay" in row:
                data["wd"].append(float(row["weight_decay"]))
            elif "dynamic_wd" in row:
                data["wd"].append(float(row["dynamic_wd"]))
            else:
                data["wd"].append(0.1)  # Default base wd
    return data


def main():
    # Load the exact data files from your latest experiments
    data_p3 = load_data("data/phase3_exp3_metrics.csv")
    data_p4a = load_data("data/phase4a_exp3_metrics.csv")
    data_p4b = load_data("data/phase4b_exp3_metrics.csv")

    # Create a 3-panel vertical layout
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    # --- PANEL 1: Phase 3 (Baseline / Control) ---
    ax1 = axes[0]
    ax1_cv = ax1.twinx()

    ax1.plot(data_p3["epoch"], data_p3["val_acc"], color='#1f77b4', lw=2, label="Validation Accuracy")
    ax1.plot(data_p3["epoch"], data_p3["train_acc"], color='#1f77b4', lw=1.5, ls='--', alpha=0.5, label="Train Accuracy")

    max_cv3 = max(data_p3["cv"]) if data_p3["cv"] else 1
    cv_norm3 = [c / max_cv3 for c in data_p3["cv"]]
    ax1_cv.plot(data_p3["epoch"], cv_norm3, color='#d62728', lw=1.5, alpha=0.8, label="$C_v$ (Normalized)")

    ax1.set_title("A) Baseline Model (Standard AdamW) — Remains trapped in memorization")
    ax1.set_ylabel("Accuracy")
    ax1_cv.set_ylabel("$C_v$ Variance", color='#d62728')
    ax1.set_ylim(-0.05, 1.05)
    ax1_cv.set_ylim(-0.05, 1.05)

    ax1.text(3800, 0.4, 'No Grokking', color='black', ha='right', va='center', fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

    # --- PANEL 2: Phase 4A (Step-Function Intervention) ---
    ax2 = axes[1]
    ax2_cv = ax2.twinx()

    ax2.plot(data_p4a["epoch"], data_p4a["val_acc"], color='#1f77b4', lw=2)
    ax2.plot(data_p4a["epoch"], data_p4a["train_acc"], color='#1f77b4', lw=1.5, ls='--', alpha=0.5)

    max_cv4a = max(data_p4a["cv"]) if data_p4a["cv"] else 1
    cv_norm4a = [c / max_cv4a for c in data_p4a["cv"]]
    ax2_cv.plot(data_p4a["epoch"], cv_norm4a, color='#d62728', lw=1.5, alpha=0.8)

    # Shade the intervention region
    wd_active = [1 if w > 0.15 else 0 for w in data_p4a["wd"]]
    ax2_cv.fill_between(data_p4a["epoch"], 0, wd_active, color='#ff7f0e', alpha=0.2, label="Heat Injection (WD=1.0)")

    # Mark Grokking
    grok_4a = 3333
    ax2.axvline(x=grok_4a, color='black', ls='-', lw=2, label=f"Grokking (Epoch {grok_4a})")

    ax2.set_title("B) Active Intervention (Step-Function) — Forced generalization via localized thermal shock")
    ax2.set_ylabel("Accuracy")
    ax2_cv.set_ylabel("$C_v$ Variance / Intervention", color='#d62728')
    ax2.set_ylim(-0.05, 1.05)
    ax2_cv.set_ylim(-0.05, 1.05)

    # --- PANEL 3: Phase 4B (Continuous CvAdamW) ---
    ax3 = axes[2]
    ax3_cv = ax3.twinx()

    ax3.plot(data_p4b["epoch"], data_p4b["val_acc"], color='#1f77b4', lw=2)
    ax3.plot(data_p4b["epoch"], data_p4b["train_acc"], color='#1f77b4', lw=1.5, ls='--', alpha=0.5)

    max_cv4b = max(data_p4b["cv"]) if data_p4b["cv"] else 1
    cv_norm4b = [c / max_cv4b for c in data_p4b["cv"]]
    ax3_cv.plot(data_p4b["epoch"], cv_norm4b, color='#d62728', lw=1.5, alpha=0.8)

    # Plot Dynamic WD
    max_wd = max(data_p4b["wd"]) if data_p4b["wd"] else 1
    wd_norm = [w / max_wd for w in data_p4b["wd"]]
    ax3_cv.plot(data_p4b["epoch"], wd_norm, color='#ff7f0e', lw=2, label="Dynamic WD (\u03BB_t)")
    ax3_cv.fill_between(data_p4b["epoch"], 0, wd_norm, color='#ff7f0e', alpha=0.15)

    # Mark Grokking
    grok_4b = 2802
    ax3.axvline(x=grok_4b, color='black', ls='-', lw=2, label=f"Grokking (Epoch {grok_4b})")

    ax3.set_title("C) Proposed $C_v$-AdamW Optimizer — Continuous thermodynamic adjustment accelerates generalization")
    ax3.set_xlabel("Training Epochs")
    ax3.set_ylabel("Accuracy")
    ax3_cv.set_ylabel("$C_v$ Variance / Dynamic WD", color='#d62728')
    ax3.set_ylim(-0.05, 1.05)
    ax3_cv.set_ylim(-0.05, 1.05)

    # --- Global Legend ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1_cv, labels1_cv = ax1_cv.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2_cv, labels2_cv = ax2_cv.get_legend_handles_labels()
    handles3_cv, labels3_cv = ax3_cv.get_legend_handles_labels()

    by_label = dict(zip(labels1 + labels1_cv + labels2 + labels2_cv + labels3_cv,
                        handles1 + handles1_cv + handles2 + handles2_cv + handles3_cv))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/Master_Publication_Figure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Master figure saved successfully to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
