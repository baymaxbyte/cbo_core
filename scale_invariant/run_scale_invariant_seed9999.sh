#!/bin/bash
# =============================================================================
# Scale-Invariant CBO Experiment
# =============================================================================
# Runs Phase 3 (z-score trigger, threshold=2) and BOTH Phase 4B variants
# (cold-start gate vs continuous sensor) across 10 seeds for 7000 epochs.
#
# Usage:
#   chmod +x run_scale_invariant.sh
#   ./run_scale_invariant.sh
# =============================================================================

set -e

DEVICE="cuda"
EPOCHS=7000
Z_THRESH=2.0
SEEDS=(9999)

echo "============================================================"
echo " SCALE-INVARIANT CBO EXPERIMENT"
echo "============================================================"
echo " Seeds: ${SEEDS[*]}"
echo " Epochs: $EPOCHS"
echo " Z-Score Threshold: $Z_THRESH"
echo " Device: $DEVICE"
echo " Variants: cold-start gate + continuous sensor (wo_coldstart)"
echo "============================================================"
echo ""

mkdir -p data figures

for SEED in "${SEEDS[@]}"; do
    
    echo ""
    echo "============================================================"
    echo " SEED $SEED — Phase 4B: CvAdamW (Cold-Start Gate)"
    echo "============================================================"
    python phase4_cv_adamw.py \
        --device $DEVICE \
        --epochs $EPOCHS \
        --z_thresh $Z_THRESH \
        --seed $SEED

    echo ""
    echo "============================================================"
    echo " SEED $SEED — Phase 4B: CvAdamW (Continuous Sensor / No Cold-Start)"
    echo "============================================================"
    python phase4_cv_adamw_wo_coldstart.py \
        --device $DEVICE \
        --epochs $EPOCHS \
        --z_thresh $Z_THRESH \
        --seed $SEED

    echo ""
    echo "  ✅ Seed $SEED complete (all 3 runs)."
    echo ""
done

echo ""
echo "============================================================"
echo " ALL 10 SEEDS COMPLETE"
echo "============================================================"
echo " Results in: data/"
echo "   phase3_seed*           — baseline triggers"
echo "   phase4b_seed*          — cold-start gate variant"
echo "   phase4b_wocs_seed*     — continuous sensor variant"
echo " Figures in: figures/"
echo "============================================================"
