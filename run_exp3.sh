#!/bin/bash
# Run Experiment 3: EMA-Smoothed Triggers + Memorization Gate
# Runs all 3 phases for 5 different seeds, 10k epochs each
# Usage: PYTHONUNBUFFERED=1 nohup bash run_exp3.sh > run_exp3.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1

SEEDS=(42 1042 2042 3042 4042)
EPOCHS=10000

echo "============================================"
echo "CBO Experiment 3: EMA Shock Absorber"
echo "  Seeds: ${SEEDS[*]}"
echo "  Epochs: $EPOCHS"
echo "============================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "############################################"
    echo "# SEED: $SEED"
    echo "############################################"

    echo ""
    echo ">>> Phase 3 (Baseline + Ghost Triggers) | seed=$SEED"
    python phase3_benchmark_exp3.py --device cuda --epochs $EPOCHS --seed $SEED

    echo ""
    echo ">>> Phase 4A (Step-Function Heat Spike) | seed=$SEED"
    python phase4_training_loop_exp3.py --device cuda --epochs $EPOCHS --seed $SEED

    echo ""
    echo ">>> Phase 4B (CvAdamW) | seed=$SEED"
    python phase4_cv_adamw_exp3.py --device cuda --epochs $EPOCHS --seed $SEED

    echo ""
    echo ">>> Plotting WD comparison (4A vs 4B) | seed=$SEED"
    python plot_wd_comparison.py --seed $SEED

    echo ""
    echo "--- Seed $SEED complete ---"
done

echo ""
echo "============================================"
echo "EXPERIMENT 3 COMPLETE (all seeds)"
echo "============================================"
