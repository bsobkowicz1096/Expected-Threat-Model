"""
Pass-only baseline with same parameters as best carry model.
For fair comparison: t=2, ctx=8, 4 layers, uniform weights, 15 epochs.
Evaluation on full pass-only val set (ctx=8 for pass-only).

Usage:
    python run_passonly_baseline.py
"""

import subprocess
import sys
import os

TRAIN_CTX = 8
TRAIN_MAX_SEQ_LEN = TRAIN_CTX + 2  # 10

# For pass-only, eval also uses ctx=8 data (pass-only sequences are shorter anyway)
# We generate ctx=8 data which serves as both train and eval
EVAL_MAX_SEQ_LEN = TRAIN_MAX_SEQ_LEN

EXPERIMENT_NAME = "xT_Phase5_PassOnly_Baseline"
VOCAB_PATH = "data/vocab_continuous_passonly.json"

TRAIN_PATH = f"data/sequences_passonly_ctx{TRAIN_CTX}_train_balanced.parquet"
VAL_PATH = f"data/sequences_passonly_ctx{TRAIN_CTX}_val_natural.parquet"

# Step 1: Generate pass-only data with ctx=8
if not os.path.exists(TRAIN_PATH):
    print(f"\n{'='*60}")
    print(f"Generating pass-only data with ctx={TRAIN_CTX}")
    print(f"{'='*60}")

    subprocess.run([
        sys.executable, "generate_passonly_data.py",
        "--max_seq_len", str(TRAIN_CTX),
    ], check=True)

# Step 2: Train pass-only model
run_name = f"passonly_ctx{TRAIN_CTX}_layers4"

cmd = [
    sys.executable, "train_xt_model.py",
    "--train_path", TRAIN_PATH,
    "--val_path", VAL_PATH,
    "--vocab_path", VOCAB_PATH,
    "--max_seq_len", str(TRAIN_MAX_SEQ_LEN),
    "--eval_max_seq_len", str(EVAL_MAX_SEQ_LEN),
    "--d_model", "512",
    "--nhead", "8",
    "--num_layers", "4",
    "--n_components", "5",
    "--dropout", "0.1",
    "--epochs", "15",
    "--batch_size", "32",
    "--lr", "1e-4",
    "--weight_decay", "0.01",
    "--weight_shot", "1.0",
    "--weight_goal", "1.0",
    "--weight_no_goal", "1.0",
    "--n_rollouts", "100",
    "--experiment_name", EXPERIMENT_NAME,
    "--run_name", run_name,
    "--notes", "Pass-only baseline: 4 layers, ctx=8, uniform weights — same params as best carry model",
]

print(f"\n{'='*60}")
print(f"Training: {run_name}")
print(f"{'='*60}")

subprocess.run(cmd, check=True)

print(f"\n{'='*60}")
print("Pass-only baseline complete!")
print(f"MLflow experiment: {EXPERIMENT_NAME}")
print(f"{'='*60}")
