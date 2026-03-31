# Expected Threat Model — Project Guide

## Current state (2026-03-28)

All 5 phases complete. Final results with 1000 Monte Carlo rollouts:

| Model | ROC-AUC | Brier | Separation |
|-------|---------|-------|------------|
| Dual MDN (carry) | 0.7586 | 0.0119 | 0.1275 |
| Single MDN (ablation) | 0.7548 | 0.0111 | 0.0789 |
| Pass-only | **0.7756** | 0.0147 | **0.1336** |

Phases 1-3 evaluated at 100 rollouts (sufficient for relative ranking of data/capacity sweeps).
Phases 4-5 re-evaluated at 1000 rollouts for stability (architectural shape comparisons).

## Thesis narrative (agreed)

Single model (dual MDN, shared vocab for pass+carry), four optimization axes + final comparison:

1. **Threshold sweep** — how much carry is noise? → t=2 wins (ROC 0.7415)
2. **Context sweep** — how much history does the model need? → ctx=8 wins (ROC 0.7494, ~65% data retention)
3. **Architecture sweep** — how much capacity? → 4 layers wins (ROC 0.7504, smaller generalizes better)
4. **Head ablation** — does a separate carry head help? → dual MDN wins on separation (+61%), ROC tied
5. **Pass-only comparison** — pass-only wins ROC (0.7756) but dual MDN has lower Brier (better calibrated)

Key insight: optimizing data (threshold, context) and capacity (layers) > loss function tuning

Narrative angle: progressive simplification — cut noisy carries, cut excess context, cut excess layers, validate dual head, then compare carry vs pass-only

## Key files

- `train_xt_model.py` — pass-only training (single MDN)
- `train_xt_model_carry_v2.py` — carry training (dual MDN)
- `generate_carry_data.py` — carry data pipeline (--carry_threshold, --max_seq_len)
- `generate_passonly_data.py` — pass-only data pipeline (--max_seq_len)
- `run_threshold_sweep.py` — Phase 1 runner
- `run_context_sweep.py` — Phase 2 runner
- `run_architecture_sweep.py` — Phase 3 runner
- `run_head_ablation.py` — Phase 4 runner (dual vs single MDN)
- `run_passonly_baseline.py` — Phase 5 runner (pass-only comparison)
- `reeval_1000_rollouts.py` — re-evaluation of Phase 4-5 models with 1000 rollouts (no retraining)
- `thesis_experiment_results.csv` — complete results table for all 16 runs across all phases
- `opis_modelu_pl.md` — detailed model description in Polish (for thesis text)
- `thesis_modifications_v2.md` — drafted thesis text changes (sections 1-4, ready to apply)

## Conventions

- Evaluation always on full val set (no truncation), separate from training data
- All experiments logged to MLflow with descriptive experiment names
- Context length = max events before terminal token; max_seq_len = context + 2 (terminal + pad)
- Phases 1-3: 100 rollouts (relative ranking); Phases 4-5: 1000 rollouts (stable comparison)
- MLflow experiments: Phase 4 = `xT_Carry_Phase4_Head_Ablation`, Phase 5 = `xT_Phase5_PassOnly_Baseline`, reeval = `xT_Phase4_5_Reeval_1000rollouts`
- User communicates in Polish, code/docs in English
