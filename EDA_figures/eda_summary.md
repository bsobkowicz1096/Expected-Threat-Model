# EDA Summary — Expected Threat Model

## Dataset

- **Source:** StatsBomb open data, 5 top European leagues (England, Spain, Germany, Italy, France), season 2015/16
- **Scale:** 6,391,338 events across 1,823 matches, grouped into 356,284 possessions

## 1. Data Completeness

Position columns relevant to the model have **zero nulls** when scoped to their event types:

| Column | Scope | Null % |
|--------|-------|--------|
| `location` | All events | 0.71% |
| `pass_end_location` | Pass events | 0.00% |
| `carry_end_location` | Carry events | 0.00% |
| `shot_outcome` | Shot events | 0.00% |

The 0.71% missing `location` values come from non-movement events (substitutions, half starts, etc.) which are filtered out in preprocessing. **No imputation needed.**

*Figure: n/a (table only)*

## 2. Event Type Distribution

Among 3,157,517 movement events used by the model:

| Type | Count | Share |
|------|-------|-------|
| Pass | 1,777,412 | 56.3% |
| Carry | 1,334,386 | 42.3% |
| Shot | 45,719 | 1.4% |

Passes dominate, carries are nearly as frequent, shots are rare (~1.4%). This extreme imbalance between movement events and terminal events (shots) is a key characteristic of the data.

*Figure: `event_type_distribution.png`*

## 3. Possession Chain Lengths

Counting only movement events (Pass, Carry, Shot) per possession:

| Statistic | Value |
|-----------|-------|
| Mean | 9.0 |
| Median | 6 |
| Std | 9.4 |
| P75 | 12 |
| P90 | 21 |
| P95 | 27 |
| Max | 185 |

The distribution is heavily right-skewed — most possessions are short (median 6 events), but a long tail extends past 50. This motivates the context length sweep (Phase 2): retaining ctx=8 captures ~65% of possessions without truncation while keeping sequences manageable for the transformer.

*Figure: `possession_chain_lengths.png`*

## 4. Shot and Goal Rates

| Metric | Value |
|--------|-------|
| Possessions ending with a shot | 42,155 (11.8%) |
| Possessions ending with a goal | 4,706 (1.32%) |
| Shot conversion rate (goals/shots) | 10.3% |

Only ~1 in 8 possessions produces a shot, and ~1 in 10 shots results in a goal. The base rate of goals (1.32%) creates severe class imbalance, motivating the balanced training set (~5% goal rate via non-goal downsampling).

*Figure: `shot_goal_waffle.png`*

## 5. Pass and Carry Distances

| | Pass | Carry |
|--|------|-------|
| Mean | 21.8 | 6.0 |
| Median | 17.5 | 3.5 |
| Std | 15.1 | 7.3 |
| Max | 121.3 | 103.6 |

**Passes** have a broad, roughly bell-shaped distribution centered around ~18 StatsBomb units (roughly 1/6 of pitch length).

**Carries** are overwhelmingly short — median 3.5 units, with 23.3% under 1 unit (essentially stationary). This motivates the carry distance threshold sweep (Phase 1): many carries are noise (player receives ball and immediately passes). Filtering at threshold t=2 removes 35.7% of carries, keeping only meaningful ball progression.

| Threshold | Carries remaining | % removed |
|-----------|-------------------|-----------|
| t=1 | 1,023,437 | 23.3% |
| t=2 | 857,909 | 35.7% |
| t=3 | 726,029 | 45.6% |
| t=5 | 531,098 | 60.2% |
| t=8 | 336,302 | 74.8% |

*Figures: `pass_distance_distribution.png`, `carry_distance_distribution.png`*

## 6. Shot Rate by Pitch Zone

Heatmap showing the percentage of possessions passing through each pitch zone that end with a shot. Clear spatial gradient: possessions touching zones near the opponent's goal have 40-60% shot probability, while those confined to the defensive half remain at 5-10%. This spatial dependency is precisely what the xT model aims to capture — the "threat" of a position on the pitch.

*Figure: `shot_rate_heatmap.png`*

## Key Takeaways for the Model

1. **Data is clean** — no missing values in model-relevant columns after filtering to movement events
2. **Severe class imbalance** (1.32% goal rate) necessitates training set balancing
3. **Short carry noise** — majority of carries are < 3 units; threshold filtering is well-motivated
4. **Right-skewed possession lengths** — context window of 8 captures most possessions without truncation
5. **Strong spatial signal** — shot probability varies dramatically by pitch zone, validating the positional approach of the xT model
