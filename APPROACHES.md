# Approach History

Chronological record of modelling approaches tried for the Expected Threat (xT) model.

---

## 1. Classic ML — XGBoost (`previous_approaches/classic.ipynb`)

**Algorithm:** XGBoost classifier with class-weight balancing (imbalanced data: ~5% goal rate).
**Data:** Italian Serie A 2015–2016, ~100K pass events.
**Features:** 14 hand-crafted features — location (x, y, end_x, end_y), pass type, height, body part, pressure, speed, etc.
**Goal:** Predict whether a pass leads to a goal within the same possession.
**Result:** ROC-AUC ~0.65, precision ~8% at 50% recall.
**Why abandoned:** Feature importance showed only raw coordinates (start_x, end_x) contributed meaningfully. Classical ML features insufficient for capturing sequence context.

---

## 2. Sequence Preparation Pipeline — PySpark (`previous_approaches/przygotowanie_zbioru.ipynb`)

Not a model — infrastructure for downstream approaches.
**Algorithm:** PySpark ETL pipeline.
**Data:** StatsBomb Serie A 2015–2016, ~1.8M events → filtered to Pass/Shot.
**Output:** 86K+ balanced tokenized sequences (START + events + GOAL/NO_GOAL), location discretised to a 5-unit grid.
**Why archived:** Successfully produced datasets but was superseded by a continuous-coordinate pipeline that doesn't require discretisation.

---

## 3. GPT-2 Discrete Token Model (`previous_approaches/continous_implement.ipynb`)

**Algorithm:** GPT-2 Small (86M parameters) fine-tuned as a causal language model on tokenized event sequences.
**Data:** ~19K balanced sequences, vocabulary of 1,272 discrete tokens (e.g. `Pass_LOC_105_35`, `Shot`, `GOAL`).
**Goal:** Predict the next token in a possession sequence; infer xT from goal-token probability.
**Result:** Training loss 5.7 → ~5.0 over 3 epochs; moderate perplexity.
**Why abandoned:** Discretising coordinates into tokens loses spatial resolution; GPT-2 overhead is high for inference-time Monte Carlo rollouts.

---

## 4. Continuous Embedding with Gaussian Output (`previous_approaches/continous_embeding_zbior-Copy1.ipynb`)

**Algorithm:** Custom PyTorch Transformer encoder + Gaussian NLL head for joint (type, x, y) prediction.
**Data:** ~19K sequences (80/20 split), max 14 events, 5% goal rate.
**Goal:** Learn joint distribution of next event type and 2D coordinates without discretisation.
**Result:** Training loss became negative after epoch 2 (instability in combined type + position losses); Monte Carlo evaluation prohibitively slow.
**Why abandoned:** Negative loss is a mathematical red flag in NLL training; switched to Mixture Density Network to handle multi-modal pass distributions.

---

## 5. MDN Concept Exploration (`previous_approaches/implementacja.ipynb`)

Theoretical / visual exploration only — no training code.
**Goal:** Show that a 5-component Gaussian mixture can represent the multi-modal distribution of pass destinations from a single field position.
**Outcome:** Motivated the use of MDN heads in the current architecture.

---

## 6. Visualisation & Encoding Reference (`veezes.ipynb`, `veezes/`)

Educational material, not a modelling approach.
**Contents:** Fourier positional encoding visualisation, event tokenisation table, MDN concept diagrams.
**Purpose:** Documentation for understanding the building blocks used in the current model.

---

## Current Approach — Transformer + MDN (`train_xt_model.py`, `train_xt_model_stage2.py`)

**Architecture:**
- Fourier position encoder for continuous (x, y, end_x, end_y) coordinates
- 12-layer Transformer encoder (d_model=512) with causal masking
- Dual output heads: event-type classification (5 classes) + Mixture Density Network for position regression
- xT score computed via Monte Carlo rollouts (100 rollouts → goal probability)

**Training pipeline:**
- Stage 1 (`run_stage1.py`): grid search over class loss weights (weight_goal, weight_shot, weight_no_goal)
- Stage 2 (`run_stage2.py`): grid search over learning rate and loss_ratio (type loss vs. MDN loss)

**Evaluation:** ROC-AUC and Brier Score on validation set; experiments tracked in MLflow.
