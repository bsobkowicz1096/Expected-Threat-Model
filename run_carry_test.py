"""
Run carry challenger experiments with baseline hyperparameters.
Tests three distance thresholds: 8, 3, 5 (in that order).
All runs land in a single MLflow experiment: xT_with_Carry_test.

Usage:
    python run_carry_test.py
"""

import subprocess
import sys

THRESHOLDS = [8, 3, 5]
EXPERIMENT_NAME = "xT_with_Carry_test"
VOCAB_PATH = "data/vocab_continuous_carry.json"

# Baseline hyperparams from Stage 1 optimization
BASELINE = {
    "d_model": 512,
    "num_layers": 12,
    "nhead": 8,
    "n_components": 5,
    "dropout": 0.1,
    "max_seq_len": 16,
    "epochs": 10,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "weight_shot": 3.0,
    "weight_goal": 15.0,
    "weight_no_goal": 1.5,
    "weight_carry": 1.0,
    "n_rollouts": 100,
}


for t in THRESHOLDS:
    train_path = f"data/sequences_carry{t}_train_balanced.parquet"
    val_path = f"data/sequences_carry{t}_val_natural.parquet"
    run_name = f"carry_threshold_{t}"

    cmd = [
        sys.executable, "train_xt_model_carry.py",
        "--train_path", train_path,
        "--val_path", val_path,
        "--vocab_path", VOCAB_PATH,
        "--experiment_name", EXPERIMENT_NAME,
        "--run_name", run_name,
        "--notes", f"Carry challenger with dist >= {t} StatsBomb units",
    ]

    for key, value in BASELINE.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Data: {train_path}")
    print(f"{'='*60}")

    subprocess.run(cmd, check=True)

print(f"\n{'='*60}")
print("All carry experiments complete!")
print(f"Check MLflow experiment: {EXPERIMENT_NAME}")
print(f"{'='*60}")
