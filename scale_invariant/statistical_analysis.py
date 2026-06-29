"""
Statistical Analysis: Baseline vs Cold-Start CvAdamW (Scale-Invariant)
======================================================================
Paired analysis across 10 seeds comparing grokking epochs.

Produces:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Cohen's d (effect size)
  - 95% Confidence Interval for mean improvement
  - Percentage improvement per seed + mean
  - Sign test (binomial)
  - Summary table for paper

Usage:
    python statistical_analysis.py
"""

import numpy as np
from scipy import stats

# ─── Data ────────────────────────────────────────────────────────────────────

seeds = [42, 123, 256, 512, 1024, 2048, 3141, 4096, 7777, 9999]

baseline = np.array([5301, 4971, 4843, 4218, 4535, 3714, 5078, 2962, 4127, 3381])
cold_start = np.array([4934, 5111, 4707, 4197, 4655, 2933, 4055, 2767, 3889, 3314])

# ─── 1. Paired Differences ──────────────────────────────────────────────────

diff = baseline - cold_start  # positive = cold-start is better (fewer epochs)

print("=" * 70)
print("PAIRED STATISTICAL ANALYSIS: Baseline vs. Cold-Start CvAdamW")
print("=" * 70)

print("\n┌────────┬──────────┬────────────┬────────────┐")
print("│  Seed  │ Baseline │ Cold-Start │ Difference │")
print("├────────┼──────────┼────────────┼────────────┤")
for s, b, c, d in zip(seeds, baseline, cold_start, diff):
    sign = "+" if d > 0 else " " if d == 0 else ""
    print(f"│  {s:<5} │   {b:>5}  │    {c:>5}   │   {sign}{d:>5}    │")
print("└────────┴──────────┴────────────┴────────────┘")

print(f"\nMean difference (d̄): {diff.mean():.1f} epochs")
print(f"Std of differences (s_d): {diff.std(ddof=1):.1f} epochs")

# ─── 2. Paired t-test ────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("PAIRED t-TEST")
print("─" * 70)
print("H₀: μ_d = 0  (no improvement)")
print("H₁: μ_d > 0  (cold-start improves grokking)")

t_stat, p_two = stats.ttest_rel(baseline, cold_start)
p_one = p_two / 2  # one-tailed (we expect improvement)

print(f"\n  t({len(diff)-1}) = {t_stat:.4f}")
print(f"  p (two-tailed) = {p_two:.4f}")
print(f"  p (one-tailed) = {p_one:.4f}")

if p_one < 0.01:
    print("  → Strong evidence against H₀ (p < 0.01)")
elif p_one < 0.05:
    print("  → Significant at α = 0.05")
else:
    print("  → Not significant at α = 0.05")

# ─── 3. Wilcoxon Signed-Rank Test ───────────────────────────────────────────

print("\n" + "─" * 70)
print("WILCOXON SIGNED-RANK TEST (nonparametric)")
print("─" * 70)

w_stat, w_p = stats.wilcoxon(baseline, cold_start, alternative='greater')
print(f"\n  W = {w_stat:.1f}")
print(f"  p (one-tailed) = {w_p:.4f}")

if w_p < 0.05:
    print("  → Significant at α = 0.05")
else:
    print("  → Not significant at α = 0.05")

# ─── 4. Effect Size (Cohen's d) ─────────────────────────────────────────────

print("\n" + "─" * 70)
print("EFFECT SIZE (Cohen's d for paired samples)")
print("─" * 70)

cohen_d = diff.mean() / diff.std(ddof=1)
print(f"\n  Cohen's d = {cohen_d:.4f}")

if abs(cohen_d) >= 1.2:
    interp = "very large"
elif abs(cohen_d) >= 0.8:
    interp = "large"
elif abs(cohen_d) >= 0.5:
    interp = "medium"
elif abs(cohen_d) >= 0.2:
    interp = "small"
else:
    interp = "negligible"
print(f"  Interpretation: {interp}")

# ─── 5. 95% Confidence Interval ─────────────────────────────────────────────

print("\n" + "─" * 70)
print("95% CONFIDENCE INTERVAL for Mean Improvement")
print("─" * 70)

mean_d = diff.mean()
sem_d = stats.sem(diff)
ci = stats.t.interval(0.95, df=len(diff)-1, loc=mean_d, scale=sem_d)

print(f"\n  Mean improvement: {mean_d:.1f} epochs")
print(f"  SEM: {sem_d:.1f}")
print(f"  95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")

if ci[0] > 0:
    print("  → CI does not include 0 — improvement is reliable")
else:
    print("  → CI includes 0 — cannot rule out no effect")

# ─── 6. Percentage Improvement ───────────────────────────────────────────────

print("\n" + "─" * 70)
print("PERCENTAGE IMPROVEMENT (per seed)")
print("─" * 70)

pct_improvement = ((baseline - cold_start) / baseline) * 100

print("\n┌────────┬──────────────────┐")
print("│  Seed  │  Improvement (%) │")
print("├────────┼──────────────────┤")
for s, p in zip(seeds, pct_improvement):
    sign = "+" if p > 0 else ""
    print(f"│  {s:<5} │     {sign}{p:>5.1f}%      │")
print("└────────┴──────────────────┘")

print(f"\n  Mean reduction: {pct_improvement.mean():.1f}% ± {pct_improvement.std(ddof=1):.1f}%")

# ─── 7. Sign Test (Binomial) ────────────────────────────────────────────────

print("\n" + "─" * 70)
print("SIGN TEST (Binomial)")
print("─" * 70)

wins = np.sum(diff > 0)
losses = np.sum(diff < 0)
ties = np.sum(diff == 0)

print(f"\n  Cold-start wins: {wins}/10")
print(f"  Cold-start loses: {losses}/10")
print(f"  Ties: {ties}/10")

sign_result = stats.binomtest(wins, wins + losses, 0.5, alternative='greater')
print(f"  Binomial test p-value: {sign_result.pvalue:.4f}")

if sign_result.pvalue < 0.05:
    print("  → Significant at α = 0.05")
else:
    print("  → Not significant at α = 0.05")

# ─── Summary Table for Paper ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY TABLE (for paper/article)")
print("=" * 70)

print(f"""
┌───────────────────────────┬──────────────────────────────────┐
│ Metric                    │ Result                           │
├───────────────────────────┼──────────────────────────────────┤
│ Mean Baseline Epoch       │ {baseline.mean():.0f}                           │
│ Mean Cold-Start Epoch     │ {cold_start.mean():.0f}                           │
│ Mean Improvement (epochs) │ {mean_d:.0f}                             │
│ Mean Improvement (%)      │ {pct_improvement.mean():.1f}%                          │
│ Paired t-test             │ t(9) = {t_stat:.3f}, p = {p_one:.4f}      │
│ Wilcoxon signed-rank      │ W = {w_stat:.0f}, p = {w_p:.4f}             │
│ Cohen's d                 │ {cohen_d:.4f} ({interp})              │
│ Wins (cold-start better)  │ {wins}/10                            │
│ Sign test p-value         │ {sign_result.pvalue:.4f}                         │
│ 95% CI (epochs)           │ [{ci[0]:.0f}, {ci[1]:.0f}]                       │
└───────────────────────────┴──────────────────────────────────┘
""")

# ─── Interpretation ──────────────────────────────────────────────────────────

print("─" * 70)
print("INTERPRETATION")
print("─" * 70)

if p_one < 0.05 and ci[0] > 0:
    print("""
The cold-start Z-score CvAdamW significantly reduces grokking latency
compared to baseline AdamW (paired t-test p < 0.05, CI excludes zero).
The method improved {}/{} seeds with a mean reduction of {:.0f} epochs
({:.1f}%), Cohen's d = {:.2f} ({}).
""".format(wins, len(seeds), mean_d, pct_improvement.mean(), cohen_d, interp))
else:
    print("""
The cold-start method improved {}/{} seeds and produced a mean reduction
of {:.0f} epochs ({:.1f}%), though statistical significance was limited
by the small number of runs (n=10). Cohen's d = {:.2f} suggests a {}
effect size. With more seeds, the effect may reach conventional
significance thresholds.
""".format(wins, len(seeds), mean_d, pct_improvement.mean(), cohen_d, interp))
