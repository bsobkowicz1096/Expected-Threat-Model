---
name: sensitivity_analysis_findings
description: Carry/Pass sensitivity analysis results and insights about token type bias in xT predictions
type: project
---

Sensitivity analysis for Carry vs Pass impact on xT (model: nofilter, weights=1.0, base=3 tokens).

**Carry (0-5 SB units):**
- Always positive delta (+0.002 at 0m to +0.012 at 5m)
- Monotonically increasing with distance
- Even zero-distance carry slightly boosts xT (model expects action after Carry)

**Pass (0-20 SB units):**
- Negative below ~5-10m (noise)
- Crosses zero around 5-10m
- Positive at 10m (+0.0016), peaks at 15m (+0.0032), slight drop at 20m (+0.0024)
- Range 0-5m was too narrow to see positive effect in first attempt

**Key finding: Pass↔Carry alternation bias**
Model learned strong alternating pattern from StatsBomb data:
- After Pass → expects Carry (55-75% probability)
- After Carry → expects Pass or Shot (depending on position)
This creates systematic bias where Carry tokens get higher contribution than equivalent-distance Pass tokens. The bias is amplified by class weights (goal=15, shot=3), but persists even with weights=1.0.

**Why:** StatsBomb data structure naturally alternates Pass/Carry. Every ball receipt is a Carry event. Model internalizes this pattern.

**How to apply:** This is a fundamental characteristic of including Carry as a separate token. It's not necessarily wrong — a player receiving the ball (Carry) and adjusting position before shooting IS more threatening than just having the ball passed nearby. But it makes Carry contributions harder to interpret independently from the alternation pattern. Consider this when designing contribution attribution.

**Methodology lessons:**
1. Must use 3-token base (not full sequence) to avoid rollout budget exhaustion
2. Model choice matters: weights=1.0 gives healthier proportions than tuned weights
3. Data/model mismatch (e.g., testing short carries on threshold_8 model) invalidates results
